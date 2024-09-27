use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use log::info;
use structopt::StructOpt;
use tokio::sync::broadcast;
use tokio::sync::Mutex;
use tokio::task::JoinSet;
use tokio::time::sleep;
use tonic::transport::Server;
use tonic::{Request, Response, Status};

use crate::torchftpb::{
    lighthouse_service_server::{LighthouseService, LighthouseServiceServer},
    LighthouseQuorumRequest, LighthouseQuorumResponse, Quorum, QuorumMember,
};

struct QuorumMemberDetails {
    joined: Instant,
    member: QuorumMember,
}

struct State {
    channel: broadcast::Sender<Quorum>,
    participants: HashMap<String, QuorumMemberDetails>,
    prev_quorum: Option<Quorum>,
}

pub struct Lighthouse {
    state: Mutex<State>,
    opt: LighthouseOpt,
}

#[derive(StructOpt, Debug)]
#[structopt()]
pub struct LighthouseOpt {
    // bind is the address to bind the server to.
    #[structopt(long = "bind", default_value = "[::]:19510")]
    bind: String,

    #[structopt(long = "join_timeout_msec", default_value = "60000")]
    join_timeout_msec: u64,

    #[structopt(long = "min_replicas")]
    min_replicas: u64,
}

impl Lighthouse {
    pub fn new(opt: LighthouseOpt) -> Arc<Self> {
        let (tx, _) = broadcast::channel(16);
        Arc::new(Self {
            state: Mutex::new(State {
                participants: HashMap::new(),
                channel: tx,
                prev_quorum: None,
            }),
            opt: opt,
        })
    }

    async fn quorum_valid(&self) -> bool {
        let state = self.state.lock().await;

        let mut first_joined = Instant::now();

        for details in state.participants.values() {
            if details.joined < first_joined {
                first_joined = details.joined;
            }
        }

        if state.prev_quorum.is_some() {
            let mut is_fast_quorum = true;
            let prev_quorum = state.prev_quorum.as_ref().unwrap();

            for prev_member in prev_quorum.participants.iter() {
                if !state.participants.contains_key(&prev_member.replica_id) {
                    is_fast_quorum = false;
                }
            }

            if is_fast_quorum {
                info!("Fast quorum found!");
                return is_fast_quorum;
            }
        }

        if state.participants.len() < self.opt.min_replicas as usize {
            info!(
                "No quorum, only have {} participants, need {}",
                state.participants.len(),
                self.opt.min_replicas
            );
            return false;
        }

        if Instant::now().duration_since(first_joined)
            < Duration::from_millis(self.opt.join_timeout_msec)
        {
            info!("No quorum, join timeout hasn't elapsed yet");
            return false;
        }

        true
    }

    pub async fn _run_quorum(self: Arc<Self>) -> Result<()> {
        loop {
            let quorum_met = self.quorum_valid().await;
            if quorum_met {
                let mut state = self.state.lock().await;
                let quorum = Quorum {
                    participants: state
                        .participants
                        .values()
                        .map(|details| details.member.clone())
                        .collect(),
                };
                state.prev_quorum = Some(quorum.clone());
                state.participants.clear();
                state.channel.send(quorum)?;
            }

            sleep(Duration::from_millis(100)).await;
        }
    }

    pub async fn _run_grpc(self: Arc<Self>) -> Result<()> {
        let bind = self.opt.bind.parse()?;
        info!("Lighthouse listening on {}", bind);

        Server::builder()
            .add_service(LighthouseServiceServer::new(self))
            .serve(bind)
            .await
            .map_err(|e| e.into())
    }

    pub async fn run(self: Arc<Self>) -> Result<()> {
        let mut set = JoinSet::new();

        set.spawn(self.clone()._run_quorum());

        set.spawn(self.clone()._run_grpc());

        while let Some(res) = set.join_next().await {
            res??;
        }
        Ok(())
    }
}

#[tonic::async_trait]
impl LighthouseService for Arc<Lighthouse> {
    async fn quorum(
        &self,
        request: Request<LighthouseQuorumRequest>,
    ) -> Result<Response<LighthouseQuorumResponse>, Status> {
        let requester = request
            .into_inner()
            .requester
            .ok_or_else(|| return Status::invalid_argument("missing requester"))?;
        let mut rx = {
            let mut state = self.state.lock().await;
            state.participants.insert(
                requester.replica_id.clone(),
                QuorumMemberDetails {
                    joined: Instant::now(),
                    member: requester,
                },
            );
            state.channel.subscribe()
        };

        let quorum = rx
            .recv()
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let reply = LighthouseQuorumResponse {
            quorum: Some(quorum),
        };

        Ok(Response::new(reply))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Sub;

    fn lighthouse_test_new() -> Arc<Lighthouse> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "n/a".to_string(),
            join_timeout_msec: 60 * 60 * 1000, // 1hr
        };
        Lighthouse::new(opt)
    }

    #[tokio::test]
    async fn test_quorum_join_timeout() {
        let lighthouse = lighthouse_test_new();
        assert!(!lighthouse.quorum_valid().await);

        {
            let mut state = lighthouse.state.lock().await;
            state.participants.insert(
                "a".to_string(),
                QuorumMemberDetails {
                    joined: Instant::now(),
                    member: QuorumMember {
                        replica_id: "a".to_string(),
                        address: "".to_string(),
                        step: 1,
                    },
                },
            );
        }

        assert!(!lighthouse.quorum_valid().await);

        {
            let mut state = lighthouse.state.lock().await;
            state.participants.get_mut("a").unwrap().joined =
                Instant::now().sub(Duration::from_secs(10 * 60 * 60));
        }

        assert!(lighthouse.quorum_valid().await);
    }

    #[tokio::test]
    async fn test_quorum_fast_prev_quorum() {
        let lighthouse = lighthouse_test_new();
        assert!(!lighthouse.quorum_valid().await);

        {
            let mut state = lighthouse.state.lock().await;
            state.participants.insert(
                "a".to_string(),
                QuorumMemberDetails {
                    joined: Instant::now(),
                    member: QuorumMember {
                        replica_id: "a".to_string(),
                        address: "".to_string(),
                        step: 1,
                    },
                },
            );
        }

        assert!(!lighthouse.quorum_valid().await);

        {
            let mut state = lighthouse.state.lock().await;
            state.prev_quorum = Some(Quorum {
                participants: vec![QuorumMember {
                    replica_id: "a".to_string(),
                    address: "".to_string(),
                    step: 1,
                }],
            });
        }

        assert!(lighthouse.quorum_valid().await);
    }
}

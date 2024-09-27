use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use log::info;
use tokio::sync::broadcast;
use tokio::sync::Mutex;
use tonic::transport::Server;
use tonic::{Request, Response, Status};

use tonic::transport::{Channel, Endpoint};

use crate::torchftpb::lighthouse_service_client::LighthouseServiceClient;
use crate::torchftpb::{
    manager_service_server::{ManagerService, ManagerServiceServer},
    LighthouseQuorumRequest, ManagerQuorumRequest, ManagerQuorumResponse, Quorum, QuorumMember,
};

struct ManagerState {
    channel: broadcast::Sender<Quorum>,
    participants: u64,
}

pub struct Manager {
    replica_id: String,
    lighthouse_addr: String,
    address: String,
    store_address: String,
    bind: String,
    world_size: u64,
    state: Mutex<ManagerState>,
}

impl Manager {
    pub fn new(
        replica_id: String,
        lighthouse_addr: String,
        address: String,
        bind: String,
        store_address: String,
        world_size: u64,
    ) -> Arc<Self> {
        let (tx, _) = broadcast::channel(16);

        Arc::new(Self {
            replica_id: replica_id,
            lighthouse_addr: lighthouse_addr,
            address: address,
            store_address: store_address,
            bind: bind,
            world_size: world_size,
            state: Mutex::new(ManagerState {
                channel: tx,
                participants: 0,
            }),
        })
    }

    pub async fn run(self: Arc<Self>) -> Result<()> {
        let bind = self.bind.parse()?;
        info!("Manager listening on {}", bind);

        Server::builder()
            .add_service(ManagerServiceServer::new(self))
            .serve(bind)
            .await
            .map_err(|e| e.into())
    }

    async fn lighthouse_client_new(&self) -> Result<LighthouseServiceClient<Channel>> {
        let conn = Endpoint::new(self.lighthouse_addr.clone())?
            .connect_timeout(Duration::from_secs(10))
            .connect()
            .await?;
        Ok(LighthouseServiceClient::new(conn))
    }
}

#[tonic::async_trait]
impl ManagerService for Arc<Manager> {
    async fn quorum(
        &self,
        request: Request<ManagerQuorumRequest>,
    ) -> Result<Response<ManagerQuorumResponse>, Status> {
        let req = request.into_inner();
        let rank = req.rank;

        info!("got quorum request for rank {}", rank);

        let mut rx = {
            let mut state = self.state.lock().await;
            // TODO check step
            state.participants += 1;
            let rx = state.channel.subscribe();

            if state.participants >= self.world_size {
                state.participants = 0;
                info!("all workers joined -- starting quorum");

                // TODO: don't hold the lock during quorum

                let mut client = self
                    .lighthouse_client_new()
                    .await
                    .map_err(|e| Status::from_error(e.into()))?;

                let request = tonic::Request::new(LighthouseQuorumRequest {
                    requester: Some(QuorumMember {
                        replica_id: self.replica_id.clone(),
                        address: self.address.clone(),
                        store_address: self.store_address.clone(),
                        step: req.step,
                    }),
                });

                let response = client.quorum(request).await.unwrap();
                info!("got lighthouse quorum");
                let resp = response.into_inner();
                state
                    .channel
                    .send(
                        resp.quorum
                            .ok_or_else(|| Status::internal("missing quorum"))?,
                    )
                    .map_err(|e| Status::from_error(e.into()))?;
            }

            rx
        };

        let quorum = rx
            .recv()
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let participants = &quorum.participants;

        let primary = &participants[rank as usize % participants.len()];
        let reply = ManagerQuorumResponse {
            quorum_id: quorum.quorum_id,
            address: primary.address.clone(),
            store_address: primary.store_address.clone(),
        };

        info!("returning quorum for rank {}", rank);

        Ok(Response::new(reply))
    }
}

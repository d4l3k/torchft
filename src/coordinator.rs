use std::time::Duration;

use anyhow::Result;
use log::{info, warn};
use raft::eraftpb::ConfChangeType;
use raft::eraftpb::ConfChangeV2;
use raft::{raw_node::RawNode, raw_node::Ready, storage::MemStorage, Config};
use slog::{o, Drain};
use tokio::time::sleep;
use tonic::{Request, Response, Status};

use crate::torchftpb::coordinator_service_server::{CoordinatorService};
use crate::torchftpb::{RaftMessageRequest, RaftMessageResponse};
use std::sync::{Arc, Mutex};

pub struct Coordinator {
    node: Mutex<RawNode<MemStorage>>,
}

impl Coordinator {
    pub fn new(rank: u64, world_size: u64) -> Self {
        let config = Config {
            // ids start at 1
            id: rank + 1,
            ..Default::default()
        };
        // Initialize logger.
        let logger = slog::Logger::root(slog_stdlog::StdLog.fuse(), o!());

        // After, make sure it's valid!
        config.validate().unwrap();

        // We don't care about the log so we can use MemStorage
        let storage = MemStorage::new_with_conf_state((vec![1], vec![]));
        let mut node = RawNode::new(&config, storage, &logger).unwrap();

        let steps = (1..world_size + 1)
            .map(|i| raft_proto::new_conf_change_single(i, ConfChangeType::AddNode))
            .collect::<Vec<_>>();
        let mut cc = ConfChangeV2::default();
        cc.set_changes(steps.into());
        node.apply_conf_change(&cc).unwrap();

        Self { node: Mutex::new(node) }
    }

    pub async fn run(&self) -> Result<()> {
        info!("running raft loop...");

        loop {
            self.tick().await?;

            // TODO: account for tick lag
            sleep(Duration::from_millis(100)).await;
        }
    }

    async fn tick(&self) -> Result<()> {
        let mut node = self.node.lock().unwrap();
        node.tick();

        if node.has_ready() {
            let ready = node.ready();

            // TODO: release node lock when processing
            self.process_ready(ready).await?;
        }

        Ok(())
    }

    async fn process_ready(&self, ready: Ready) -> Result<()> {
        Ok(())
    }
}

#[tonic::async_trait]
impl CoordinatorService for Arc<Coordinator> {
    async fn raft_message(
        &self,
        request: Request<RaftMessageRequest>, // Accept request of type HelloRequest
    ) -> Result<Response<RaftMessageResponse>, Status> {
        // Return an instance of type HelloReply
        println!("Got a request: {:?}", request);

        let reply = RaftMessageResponse {
            message: format!("Hello {}!", request.into_inner().name), // We must use .into_inner() as the fields of gRPC requests and responses are private
        };

        Ok(Response::new(reply)) // Send back our formatted greeting
    }
}

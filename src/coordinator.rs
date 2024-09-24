use std::time::Duration;

use anyhow::Result;
use log::{info, warn};
use raft::{raw_node::RawNode, storage::MemStorage, Config};
use slog::{o, Drain};
use tokio::time::sleep;

pub struct Coordinator {
    node: RawNode<MemStorage>,
}

impl Coordinator {
    pub fn new() -> Self {
        let config = Config {
            id: 1,
            ..Default::default()
        };
        // Initialize logger.
        let logger = slog::Logger::root(slog_stdlog::StdLog.fuse(), o!());

        // After, make sure it's valid!
        config.validate().unwrap();

        // We don't care about the log so we can use MemStorage
        let storage = MemStorage::new_with_conf_state((vec![1], vec![]));
        let node = RawNode::new(&config, storage, &logger).unwrap();

        Self { node: node }
    }

    pub async fn run(&mut self) -> Result<()> {
        info!("running...");

        loop {
            self.node.tick();

            // TODO: account for tick lag
            sleep(Duration::from_millis(100)).await;
        }
    }
}

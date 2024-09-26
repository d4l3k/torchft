use core::net::SocketAddr;
use std::sync::Arc;

use anyhow::Result;
use log::info;
use structopt::StructOpt;
use tokio::runtime::Runtime;
use tokio::task::JoinSet;
use tonic::transport::Server;

use torchft::coordinator::Coordinator;

use torchft::torchftpb::coordinator_service_server::CoordinatorServiceServer;

/// A StructOpt example
#[derive(StructOpt, Debug)]
#[structopt()]
struct Opt {
    // bind is the address to bind the server to.
    #[structopt(long = "bind", default_value = "[::]:50051")]
    bind: String,
    #[structopt(long = "rank", default_value = "0")]
    rank: u64,
    #[structopt(long = "world_size", default_value = "3")]
    world_size: u64,

    #[structopt(long = "bootstrap", default_value = "")]
    bootstrap: Vec<String>,
}

fn main() {
    stderrlog::new()
        .verbosity(2)
        .show_module_names(true)
        .timestamp(stderrlog::Timestamp::Millisecond)
        .init()
        .unwrap();

    let rt = Runtime::new().unwrap();

    rt.block_on(main_async()).unwrap();
}

async fn main_async() -> Result<()> {
    let opt = Opt::from_args();
    let bind: SocketAddr = opt.bind.parse().unwrap();

    let local_addr = format!("http://{}:{}", "localhost", bind.port());

    let c = Arc::new(Coordinator::new(
        opt.rank,
        opt.world_size,
        local_addr.clone(),
    ));

    info!(
        "Coordinator listening on {}, local_addr={}",
        bind, local_addr
    );

    let mut set: JoinSet<Result<()>> = JoinSet::new();

    set.spawn(c.clone().run());

    let c_grpc = c.clone();
    set.spawn(async move {
        Server::builder()
            .add_service(CoordinatorServiceServer::new(c_grpc))
            .serve(bind)
            .await
            .map_err(|e| e.into())
    });
    set.spawn(c.clone().bootstrap(opt.bootstrap));

    while let Some(res) = set.join_next().await {
        res??;
    }

    Ok(())
}

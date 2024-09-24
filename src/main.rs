use std::sync::Arc;
use std::fmt::format;
use core::net::SocketAddr;

use log::info;
use structopt::StructOpt;
use tokio::runtime::Runtime;
use tonic::transport::Server;
use anyhow::Result;

mod coordinator;
use coordinator::Coordinator;

use torchftpb::coordinator_service_server::CoordinatorServiceServer;

pub mod torchftpb {
    tonic::include_proto!("torchft");
}

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


    rt.block_on(main_async(&rt)).unwrap();
}

async fn main_async(rt: &Runtime) -> Result<()> {
    let opt = Opt::from_args();
    let bind: SocketAddr = opt.bind.parse().unwrap();

    let local_addr = format!("{}:{}", "localhost", bind.port());

    let c = Arc::new(Coordinator::new(opt.rank, opt.world_size, local_addr.clone()));


    info!("Coordinator listening on {}, local_addr={}", bind, local_addr);

    let raft_handle = rt.spawn(c.clone().run());
    let grpc_handle = rt.spawn(
        Server::builder()
            .add_service(CoordinatorServiceServer::new(c.clone()))
            .serve(bind),
    );
    let bootstrap_handle = rt.spawn(c.clone().bootstrap(opt.bootstrap));

    raft_handle.await?;
    grpc_handle.await?;
    bootstrap_handle.await?;

    Ok(())
}

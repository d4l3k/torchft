use std::sync::Arc;

use log::info;
use structopt::StructOpt;
use tokio::runtime::Runtime;
use tonic::transport::Server;

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
    #[structopt(long = "bind", default_value = "[::1]:50051")]
    bind: String,
    #[structopt(long = "rank", default_value = "0")]
    rank: u64,
    #[structopt(long = "world_size", default_value = "3")]
    world_size: u64,

    #[structopt(long = "bootstrap", default_value = "localhost:50051")]
    bootstrap: Vec<String>,
}

fn main() {
    stderrlog::new()
        .verbosity(2)
        .show_module_names(true)
        .timestamp(stderrlog::Timestamp::Millisecond)
        .init()
        .unwrap();

    let opt = Opt::from_args();

    let c = Arc::new(Coordinator::new(opt.rank, opt.world_size));
    let rt = Runtime::new().unwrap();

    let addr = opt.bind.parse().unwrap();

    info!("Coordinator listening on {}", addr);

    let grpc_handle = rt.spawn(
        Server::builder()
            .add_service(CoordinatorServiceServer::new(c.clone()))
            .serve(addr),
    );

    rt.block_on(c.run()).unwrap();
    rt.block_on(grpc_handle).unwrap();
}

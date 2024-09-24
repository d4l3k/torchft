use std::sync::Arc;

use tokio::runtime::Runtime;

mod coordinator;
use coordinator::Coordinator;

use tonic::transport::Server;
use torchftpb::coordinator_service_server::{CoordinatorServiceServer};


pub mod torchftpb {
    tonic::include_proto!("torchft");
}

fn main() {
    stderrlog::new()
        .verbosity(2)
        .show_module_names(true)
        .timestamp(stderrlog::Timestamp::Millisecond)
        .init()
        .unwrap();

    let c = Arc::new(Coordinator::new(0, 3));
    let rt = Runtime::new().unwrap();

    let addr = "[::1]:50051".parse().unwrap();

    let grpc_handle = rt.spawn(
        Server::builder()
            .add_service(CoordinatorServiceServer::new(c.clone()))
            .serve(addr),
    );

    rt.block_on(c.run()).unwrap();
    rt.block_on(grpc_handle).unwrap();
}

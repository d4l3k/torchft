use tokio::runtime::Runtime;

mod coordinator;
use coordinator::Coordinator;

use tonic::{transport::Server, Request, Response, Status};
use torchftpb::coordinator_service_server::{CoordinatorService, CoordinatorServiceServer};

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

    let mut c = Coordinator::new(0, 3);
    let rt = Runtime::new().unwrap();

    let addr = "[::1]:50051".parse().unwrap();

    let grpcHandle = rt.spawn(
        Server::builder()
            .add_service(CoordinatorServiceServer::new(c))
            .serve(addr),
    );

    rt.block_on(c.run()).unwrap();
    rt.block_on(grpcHandle);
}

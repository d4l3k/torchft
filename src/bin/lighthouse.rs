use std::sync::Arc;
use structopt::StructOpt;
use torchft::lighthouse::{Lighthouse, LighthouseOpt};

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    stderrlog::new()
        .verbosity(2)
        .show_module_names(true)
        .timestamp(stderrlog::Timestamp::Millisecond)
        .init()
        .unwrap();

    let opt = LighthouseOpt::from_args();
    let lighthouse = Arc::new(Lighthouse::new(opt));

    lighthouse.run().await.unwrap();
}

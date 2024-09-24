use tokio::runtime::Runtime;

mod coordinator;
use coordinator::Coordinator;

fn main() {
    stderrlog::new()
        .verbosity(2)
        .show_module_names(true)
        .timestamp(stderrlog::Timestamp::Millisecond)
        .init()
        .unwrap();

    let mut c = Coordinator::new();
    let rt = Runtime::new().unwrap();
    rt.block_on(c.run()).unwrap()
}

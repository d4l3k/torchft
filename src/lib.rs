pub mod coordinator;
pub mod lighthouse;
pub mod manager;

use std::sync::Arc;

use anyhow::Result;
use tokio::runtime::Runtime;
use tokio::task::JoinHandle;

pub mod torchftpb {
    tonic::include_proto!("torchft");
}

use pyo3::prelude::*;

#[pyclass]
struct Manager {
    runtime: Runtime,
    manager: Arc<manager::Manager>,
    handle: JoinHandle<Result<()>>,
}

#[pymethods]
impl Manager {
    #[new]
    fn new(
        replica_id: String,
        lighthouse_addr: String,
        address: String,
        bind: String,
        store_addr: String,
        world_size: u64,
    ) -> Self {
        let runtime = Runtime::new().unwrap();
        let manager = manager::Manager::new(
            replica_id,
            lighthouse_addr,
            address,
            bind,
            store_addr,
            world_size,
        );
        let handle = runtime.spawn(manager.clone().run());
        Self {
            runtime: runtime,
            manager: manager,
            handle: handle,
        }
    }
}

#[pyclass]
struct ManagerClient {}

#[pymethods]
impl ManagerClient {
    #[new]
    fn new(addr: String) -> Self {
        Self {}
    }
}

#[pymodule]
fn torchft(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // setup logging on import
    stderrlog::new()
        .verbosity(2)
        .show_module_names(true)
        .timestamp(stderrlog::Timestamp::Millisecond)
        .init()
        .unwrap();

    m.add_class::<Manager>()?;
    m.add_class::<ManagerClient>()?;

    Ok(())
}

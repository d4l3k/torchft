pub mod coordinator;
pub mod lighthouse;
pub mod manager;

use std::sync::Arc;

use anyhow::Result;
use pyo3::exceptions::PyRuntimeError;
use tokio::runtime::Runtime;
use tokio::task::JoinHandle;
use tonic::transport::Channel;

pub mod torchftpb {
    tonic::include_proto!("torchft");
}

use crate::torchftpb::manager_service_client::ManagerServiceClient;
use crate::torchftpb::{CheckpointAddressRequest, ManagerQuorumRequest};
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
        py: Python<'_>,
        replica_id: String,
        lighthouse_addr: String,
        address: String,
        bind: String,
        store_addr: String,
        world_size: u64,
    ) -> Self {
        py.allow_threads(move || {
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
        })
    }

    fn shutdown(&self, py: Python<'_>) {
        py.allow_threads(move || {
            self.handle.abort();
        })
    }
}

#[pyclass]
struct ManagerClient {
    runtime: Runtime,
    client: ManagerServiceClient<Channel>,
}

#[pymethods]
impl ManagerClient {
    #[new]
    fn new(py: Python<'_>, addr: String) -> PyResult<Self> {
        py.allow_threads(move || {
            let runtime = Runtime::new().unwrap();
            let client = runtime
                .block_on(manager::manager_client_new(addr))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            Ok(Self {
                runtime: runtime,
                client: client,
            })
        })
    }

    fn quorum(
        &mut self,
        py: Python<'_>,
        rank: i64,
        step: i64,
        checkpoint_server_addr: String,
    ) -> PyResult<(i64, i64, i64, String, String, i64, i64)> {
        py.allow_threads(move || {
            let request = tonic::Request::new(ManagerQuorumRequest {
                rank: rank,
                step: step,
                checkpoint_server_addr: checkpoint_server_addr,
            });
            let response = self
                .runtime
                .block_on(self.client.quorum(request))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let resp = response.into_inner();
            Ok((
                resp.quorum_id,
                resp.replica_rank,
                resp.replica_world,
                resp.address,
                resp.store_address,
                resp.max_step,
                resp.num_max,
            ))
        })
    }

    fn checkpoint_address(&mut self, py: Python<'_>, rank: i64) -> PyResult<String> {
        py.allow_threads(move || {
            let request = tonic::Request::new(CheckpointAddressRequest { rank: rank });
            let response = self
                .runtime
                .block_on(self.client.checkpoint_address(request))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let resp = response.into_inner();
            Ok(resp.checkpoint_server_address)
        })
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

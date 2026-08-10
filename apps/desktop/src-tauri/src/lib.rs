// Narrow desktop/system bridge for LLMForeman.
//
// This layer is intentionally minimal: no product/domain logic, no IPC
// protocol, no Python sidecar management. Orchestration lives in the Python
// runtime, not here.

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

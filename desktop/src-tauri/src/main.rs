// No console window behind Fritz in a release build.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    fritz_desktop_lib::run();
}

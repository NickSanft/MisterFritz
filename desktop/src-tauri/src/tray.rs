//! The tray: Show / Hide Fritz, Arrange / Lock, the spike's log folder, Quit.
//! The menu is rebuilt when Arrange starts or ends (the machine's
//! `RefreshTray`) and when Fritz is parked or shown, since its items follow
//! both. Show and Hide are separate ids, so a click does what the label it
//! showed said, whatever the state is by the time it lands.

use tauri::menu::{Menu, MenuBuilder, MenuEvent};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Manager};

use crate::overlay_machine::{ModeRequest, TRAY_LOCK_ID, TRAY_LOCK_LABEL};
use crate::overlay_windows;

const TRAY_ID: &str = "fritz";
const SHOW: &str = "show";
const HIDE: &str = "hide";
const ARRANGE: &str = "arrange";
const LOGS: &str = "logs";
const QUIT: &str = "quit";

/// The state-dependent items: (id, label) for Show / Hide, and for Arrange /
/// Lock when one is on offer. Arrange is only offered while he is on screen,
/// since the machine refuses it while he is hidden or gone.
pub fn items(
    open: bool,
    parked: bool,
    arranging: bool,
) -> (
    (&'static str, &'static str),
    Option<(&'static str, &'static str)>,
) {
    let shown = open && !parked;
    let show_hide = if shown {
        (HIDE, "Hide Fritz")
    } else {
        (SHOW, "Show Fritz")
    };
    let arrange = if arranging {
        Some((TRAY_LOCK_ID, TRAY_LOCK_LABEL))
    } else if shown {
        Some((ARRANGE, "Arrange Fritz…"))
    } else {
        None
    };
    (show_hide, arrange)
}

fn menu(app: &AppHandle) -> tauri::Result<Menu<tauri::Wry>> {
    let status = overlay_windows::status(app);
    let ((sh_id, sh_label), arrange) =
        items(status.open, status.parked, overlay_windows::arranging(app));
    let mut m = MenuBuilder::new(app).text(sh_id, sh_label);
    if let Some((id, label)) = arrange {
        m = m.text(id, label);
    }
    m.separator()
        .text(LOGS, "Open the spike logs")
        .separator()
        .text(QUIT, "Quit Mister Fritz")
        .build()
}

pub fn build(app: &AppHandle) -> tauri::Result<()> {
    let mut tray = TrayIconBuilder::with_id(TRAY_ID)
        .tooltip("Mister Fritz")
        .menu(&menu(app)?)
        .show_menu_on_left_click(true)
        .on_menu_event(on_menu);
    if let Some(icon) = app.default_window_icon() {
        tray = tray.icon(icon.clone());
    }
    tray.build(app)?;
    Ok(())
}

/// Rebuild the menu for the current state. Main thread.
pub fn refresh(app: &AppHandle) {
    let Some(tray) = app.tray_by_id(TRAY_ID) else {
        return;
    };
    match menu(app) {
        Ok(m) => {
            let _ = tray.set_menu(Some(m));
        }
        Err(e) => tracing::warn!(error = %e, "couldn't rebuild the tray menu"),
    }
}

fn on_menu(app: &AppHandle, event: MenuEvent) {
    match event.id().as_ref() {
        SHOW => crate::show_avatar(app),
        HIDE => {
            overlay_windows::set_parked(app, true);
        }
        ARRANGE => {
            if let Err(e) = overlay_windows::set_mode(app, ModeRequest::Arrange) {
                tracing::info!(error = %e, "arrange refused");
            }
        }
        TRAY_LOCK_ID => {
            let _ = overlay_windows::set_mode(app, ModeRequest::Locked);
        }
        LOGS => {
            if let Ok(dir) = app.path().app_data_dir() {
                let _ = std::process::Command::new("explorer").arg(dir).spawn();
            }
        }
        QUIT => crate::quit(app),
        _ => {}
    }
    refresh(app);
}

#[cfg(test)]
mod tests {
    use super::{items, ARRANGE, HIDE, SHOW};
    use crate::overlay_machine::{TRAY_LOCK_ID, TRAY_LOCK_LABEL};

    #[test]
    fn the_menu_says_what_a_click_does() {
        assert_eq!(items(true, false, false).0, (HIDE, "Hide Fritz"));
        assert_eq!(items(true, true, false).0, (SHOW, "Show Fritz"));
        // Closed (Alt+F4, or the watchdog): Show unparks and reopens him.
        assert_eq!(items(false, false, false).0, (SHOW, "Show Fritz"));
        assert_eq!(items(false, true, false).0, (SHOW, "Show Fritz"));
    }

    /// Arrange only while he is on screen (the machine refuses it otherwise);
    /// Lock whenever arranging.
    #[test]
    fn arrange_is_offered_only_when_it_would_work() {
        assert_eq!(items(true, false, false).1.map(|i| i.0), Some(ARRANGE));
        assert_eq!(items(true, true, false).1, None, "hidden");
        assert_eq!(items(false, false, false).1, None, "closed");
        assert_eq!(
            items(true, false, true).1,
            Some((TRAY_LOCK_ID, TRAY_LOCK_LABEL))
        );
    }
}

"""Orthogonal geometry tracking for UI widgets.

ORTHOGONAL ARCHITECTURE:
- WidgetSizeMonitor: Only detects size changes
- AutoGeometryTracker: Only discovers relevant widgets
- Each abstraction solves one problem completely, generically, and composably

This module provides reusable geometry tracking that can be used by any system
that needs to react to widget size changes (flash overlays, layout managers, etc.)
"""

import logging
from typing import Callable, List, Set, Dict, Optional
from PyQt6.QtCore import QEvent, QObject
from PyQt6.QtWidgets import QWidget

logger = logging.getLogger(__name__)


class WidgetSizeMonitor(QObject):
    """Monitors widget size changes and provides notifications.
    
    SINGLE RESPONSIBILITY: Only detects size changes in watched widgets.
    Provides a clean callback interface for systems that need to react to size changes.
    """
    
    def __init__(self):
        super().__init__()
        self._size_changed_callbacks: List[Callable[[QWidget], None]] = []
        self._watched_widgets: Set[int] = set()
        
    def watch_widget(self, widget: QWidget) -> None:
        """Watch a widget for size changes.
        
        Args:
            widget: The widget to monitor for size changes
        """
        widget_id = id(widget)
        if widget_id not in self._watched_widgets:
            self._watched_widgets.add(widget_id)
            widget.installEventFilter(self)
            logger.debug(f"[GEOMETRY] Watching widget {widget.__class__.__name__} for size changes")
            
    def unwatch_widget(self, widget: QWidget) -> None:
        """Stop watching a widget for size changes.
        
        Args:
            widget: The widget to stop monitoring
        """
        widget_id = id(widget)
        if widget_id in self._watched_widgets:
            self._watched_widgets.remove(widget_id)
            # Note: We don't remove the event filter as it may be shared
            
    def on_size_changed(self, callback: Callable[[QWidget], None]) -> None:
        """Register callback for when any watched widget changes size.
        
        Args:
            callback: Function that receives the widget that changed size
        """
        self._size_changed_callbacks.append(callback)
        
    def eventFilter(self, obj: QWidget, event: QEvent) -> bool:
        """Detect size changes in watched widgets.
        
        Args:
            obj: The widget being monitored
            event: The Qt event
            
        Returns:
            True if event was handled, False otherwise
        """
        if id(obj) not in self._watched_widgets:
            return super().eventFilter(obj, event)
            
        if event.type() == QEvent.Type.Resize:
            current_size = obj.size()
            
            # Check if size actually changed
            previous_size = getattr(obj, '_monitored_size', None)
            if previous_size is None or current_size != previous_size:
                # Store new size for next comparison
                obj._monitored_size = current_size
                
                logger.debug(f"[GEOMETRY] Size changed in {obj.__class__.__name__}: {previous_size} → {current_size}")
                
                # Notify all callbacks - FAIL LOUD if callback fails
                for callback in self._size_changed_callbacks:
                    callback(obj)
                
        return super().eventFilter(obj, event)


class AutoGeometryTracker:
    """Automatically discovers and tracks all geometry-affecting widgets.
    
    SINGLE RESPONSIBILITY: Only discovers widgets that could affect geometry.
    Provides automatic widget discovery and monitoring without manual registration.
    """
    
    def __init__(self, window: QWidget, monitor: WidgetSizeMonitor):
        """Initialize the auto geometry tracker.
        
        Args:
            window: The window containing widgets to track
            monitor: The size monitor to use for tracking
        """
        self._window = window
        self._monitor = monitor
        
        # Discover and watch all geometry-affecting widgets
        self._discover_geometry_widgets()
        
        # Listen for size changes and notify interested systems
        self._monitor.on_size_changed(self._on_widget_size_changed)
        
    def _discover_geometry_widgets(self) -> None:
        """Discover all widgets that could affect flash geometry.
        
        Watches:
        - QLabel: For dirty markers, titles, and text changes
        - QGroupBox: For flash target groupboxes
        - QAbstractItemView: For list/tree widgets that contain flash sources
        """
        from PyQt6.QtWidgets import QLabel, QGroupBox, QAbstractItemView
        
        # Track all labels (dirty markers, titles, etc.)
        labels = self._window.findChildren(QLabel)
        for label in labels:
            self._monitor.watch_widget(label)
            
        # Track all groupboxes (flash targets)
        groupboxes = self._window.findChildren(QGroupBox)
        for groupbox in groupboxes:
            self._monitor.watch_widget(groupbox)
            
        # Track all list/tree widgets (flash sources)
        list_widgets = self._window.findChildren(QAbstractItemView)
        for list_widget in list_widgets:
            self._monitor.watch_widget(list_widget)
            
        logger.info(f"[GEOMETRY] Auto-discovered and watching: {len(labels)} labels, "
                   f"{len(groupboxes)} groupboxes, {len(list_widgets)} list/tree widgets")
        
    def _on_widget_size_changed(self, widget: QWidget) -> None:
        """React to any geometry-affecting widget changing size.
        
        This method can be overridden by subclasses to provide custom behavior.
        Default implementation just logs the change.
        
        Args:
            widget: The widget that changed size
        """
        logger.debug(f"[GEOMETRY] Auto-detected size change in {widget.__class__.__name__}")


class FlashGeometryTracker(AutoGeometryTracker):
    """Specialized geometry tracker for flash overlay system.
    
    ORTHOGONAL APPROACH: Eliminates timing complexity rather than managing it.
    
    FUNDAMENTAL PRINCIPLE: Never start flashes immediately when size changes occur.
    All flash requests are queued until layout state becomes stable through explicit
    state transitions, not arbitrary timing values.
    
    This prevents the race condition entirely by changing WHEN flashes can start,
    not trying to guess WHEN layout operations complete.
    """
    
    def __init__(self, window: QWidget, monitor: WidgetSizeMonitor, flash_overlay: Optional[QWidget] = None):
        """Initialize flash geometry tracker.
        
        Args:
            window: The window containing widgets to track
            monitor: The size monitor to use for tracking
            flash_overlay: The flash overlay to invalidate when geometry changes
        """
        super().__init__(window, monitor)
        self._flash_overlay = flash_overlay
        self._layout_unstable = False
        self._queued_flashes: List[Callable[[], None]] = []
        
    def set_flash_overlay(self, flash_overlay: QWidget) -> None:
        """Set or update the flash overlay to invalidate.
        
        Args:
            flash_overlay: The flash overlay to invalidate when geometry changes
        """
        self._flash_overlay = flash_overlay
        
    def queue_flash_until_layout_stable(self, flash_callable: Callable[[], None]) -> None:
        """Queue a flash request to be processed when layout is stable.
        
        ORTHOGONAL PRINCIPLE: Flash behavior is declarative based on layout state,
        not timing. When layout is unstable, flashes are ALWAYS queued.
        When layout is stable, flashes can start immediately.
        
        Args:
            flash_callable: Function that will start the flash when called
        """
        if self._layout_unstable:
            # Layout is unstable - ALWAYS queue the flash (no exceptions)
            queue_size_before = len(self._queued_flashes)
            self._queued_flashes.append(flash_callable)
            queue_size_after = len(self._queued_flashes)
            
            logger.info(f"[FLASH] ⏳ ALWAYS QUEUED flash until layout stable (queued: {queue_size_before} -> {queue_size_after})")
        else:
            # Layout is stable - start flash immediately
            logger.info(f"[FLASH] ⚡ Layout is stable, starting flash immediately")
            flash_callable()
            
    def mark_layout_unstable(self) -> None:
        """Mark layout as unstable - future flashes will be queued.
        
        This should be called when we know layout operations are starting.
        """
        if not self._layout_unstable:
            self._layout_unstable = True
            logger.info(f"[FLASH] Layout marked as UNSTABLE")
        else:
            logger.debug(f"[FLASH] Layout already unstable")
            
    def mark_layout_stable_and_process_queued_flashes(self) -> None:
        """Mark layout as stable and process all queued flash requests.
        
        This should be called when we know layout operations have completed.
        This is the ONLY way flashes start when layout was previously unstable.
        """
        was_unstable = self._layout_unstable
        queued_count = len(self._queued_flashes)
        
        if was_unstable:
            logger.info(f"[FLASH] Layout marked as STABLE - processing {queued_count} queued flashes")
            
            # Invalidate cache first (before starting flashes) - FAIL LOUD if this fails
            if self._flash_overlay is not None:
                from .flash_mixin import WindowFlashOverlay
                overlay = WindowFlashOverlay.get_for_window(self._window)
                if overlay:
                    overlay.invalidate_cache()
                    logger.info(f"[FLASH] ✅ Invalidated cache after layout completion")
                else:
                    logger.warning(f"[FLASH] ⚠️ No overlay found for window during cache invalidation")
            
            # Process queued flashes (now with stable geometry) - FAIL LOUD if this fails
            if queued_count > 0:
                logger.info(f"[FLASH] 🔥 Processing {queued_count} queued flash requests...")
                for i, flash_callable in enumerate(self._queued_flashes):
                    try:
                        flash_callable()
                        logger.info(f"[FLASH] ✅ Started queued flash {i+1}/{queued_count}")
                    except Exception as e:
                        logger.error(f"[FLASH] ❌ Failed to start queued flash {i+1}/{queued_count}: {e}")
                        raise  # Re-raise to fail loud
                
                self._queued_flashes.clear()
                logger.info(f"[FLASH] ✅ Processed {queued_count} queued flash requests")
            else:
                logger.info(f"[FLASH] Layout completed but no flashes were queued")
            
            # Mark layout as stable
            self._layout_unstable = False
            logger.info(f"[FLASH] Layout marked as STABLE")
        else:
            logger.debug(f"[FLASH] Layout was already stable, no action needed")
        
    def _on_widget_size_changed(self, widget: QWidget) -> None:
        """React to widget size changes by marking layout as unstable.
        
        Automatically detects when layout completes and processes any pending flashes.
        
        Args:
            widget: The widget that changed size
        """
        # Call parent for logging
        super()._on_widget_size_changed(widget)
        
        # ALWAYS mark layout as unstable when size changes
        # This prevents any flashes from starting until layout completes
        self.mark_layout_unstable()
        
        # Use Qt's coalesced event handling to detect when layout operations complete
        # Qt processes all pending events before returning control, so a single-shot timer
        # with 0ms delay will fire AFTER all layout changes have been processed
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, self._on_layout_operations_complete)
        
    def _on_layout_operations_complete(self) -> None:
        """Called after all pending layout operations have been processed.
        
        This uses Qt's event loop to detect when layout changes are complete.
        A single-shot timer with 0ms delay will fire after Qt has processed
        all pending resize/layout events, giving us deterministic completion detection.
        """
        # Mark layout as stable and process any queued flashes
        self.mark_layout_stable_and_process_queued_flashes()


# Convenience function for easy integration
def create_flash_geometry_tracking(window: QWidget, flash_overlay: Optional[QWidget] = None) -> FlashGeometryTracker:
    """Create flash geometry tracking for a window.
    
    Args:
        window: The window to track geometry changes for
        flash_overlay: Optional flash overlay to invalidate on size changes
        
    Returns:
        FlashGeometryTracker instance ready to use
    """
    monitor = WidgetSizeMonitor()
    return FlashGeometryTracker(window, monitor, flash_overlay)
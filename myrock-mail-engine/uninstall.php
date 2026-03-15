<?php
/**
 * Uninstall handler for MyRock Mail Engine.
 *
 * WordPress calls this file when the plugin is deleted from the Plugins screen.
 * It must be a standalone file — the plugin is not loaded at this point, so
 * we bootstrap only what we need.
 *
 * Data removal is conditional on the `mrme_delete_on_uninstall` option so site
 * owners can choose to preserve their data.
 *
 * @package MyRock\MailEngine
 */

// Guard: this file must only be executed by WordPress during plugin deletion.
if ( ! defined( 'WP_UNINSTALL_PLUGIN' ) ) {
    exit;
}

/*
 * We need the plugin's Activator class for its static uninstall() method.
 * Use plugin_dir_path relative to THIS file (not __DIR__ alone, which is the
 * same, but being explicit avoids confusion).
 */
$plugin_main_file = plugin_dir_path( __FILE__ ) . 'myrock-mail-engine.php';

if ( file_exists( $plugin_main_file ) ) {
    /*
     * Including the main plugin file bootstraps autoloading so that the
     * MyRock\MailEngine\Core\Activator class becomes available.
     * The main file must guard against double-execution itself.
     */
    require_once $plugin_main_file;
}

// Call the static uninstall routine if the class exists.
if ( class_exists( 'MyRock\\MailEngine\\Core\\Activator' ) ) {
    \MyRock\MailEngine\Core\Activator::uninstall();
}

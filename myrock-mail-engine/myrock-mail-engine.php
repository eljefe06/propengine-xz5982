<?php
/**
 * Plugin Name:       MyRock Mail Engine
 * Plugin URI:        https://myrock.com.mx/plugin/
 * Description:       Email marketing y gestión de contactos profesional para WordPress. Standalone, portable y listo para escalar.
 * Version:           1.0.4
 * Requires at least: 6.0
 * Requires PHP:      8.0
 * Author:            MyRock
 * Author URI:        https://myrock.com.mx
 * License:           GPLv2 or later
 * License URI:       https://www.gnu.org/licenses/gpl-2.0.html
 * Text Domain:       myrock-mail-engine
 * Domain Path:       /languages
 *
 * @package MyRock\MailEngine
 */

defined( 'ABSPATH' ) || exit;

// Plugin constants
define( 'MRME_VERSION',     '1.0.4' );
define( 'MRME_DB_VERSION',  '1.0.4' );
define( 'MRME_FILE',        __FILE__ );
define( 'MRME_DIR',         plugin_dir_path( __FILE__ ) );
define( 'MRME_URL',         plugin_dir_url( __FILE__ ) );
define( 'MRME_BASENAME',    plugin_basename( __FILE__ ) );

// Autoloader — simple PSR-4 style
spl_autoload_register( function ( string $class ) {
	$prefix = 'MyRock\\MailEngine\\';
	if ( ! str_starts_with( $class, $prefix ) ) return;

	$relative = str_replace( '\\', DIRECTORY_SEPARATOR, substr( $class, strlen( $prefix ) ) );
	$file      = MRME_DIR . 'includes' . DIRECTORY_SEPARATOR . $relative . '.php';

	if ( file_exists( $file ) ) {
		require_once $file;
	}
} );

// Activation / Deactivation
register_activation_hook(   __FILE__, [ 'MyRock\\MailEngine\\Core\\Activator',   'activate'   ] );
register_deactivation_hook( __FILE__, [ 'MyRock\\MailEngine\\Core\\Deactivator', 'deactivate' ] );
register_uninstall_hook(    __FILE__, [ 'MyRock\\MailEngine\\Core\\Activator',   'uninstall'  ] );

// Boot the plugin
add_action( 'plugins_loaded', function () {
	MyRock\MailEngine\Core\Plugin::instance()->run();
} );

/**
 * Helper: public API for shortcode
 */
function mrme_form_shortcode( $atts ) {
	return MyRock\MailEngine\Public\Shortcodes::render_form( $atts );
}

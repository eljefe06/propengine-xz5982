<?php
/**
 * Plugin Name:       MyRock License Server
 * Plugin URI:        https://myrock.com.mx/plugin
 * Description:       Servidor de licencias para MyRock Mail Engine. Genera y valida llaves Pro con cobro recurrente vía MercadoPago.
 * Version:           1.0.0
 * Requires at least: 6.0
 * Requires PHP:      8.0
 * Author:            MyRock
 * Author URI:        https://myrock.com.mx
 * License:           GPL-2.0-or-later
 * Text Domain:       mrls
 */

defined( 'ABSPATH' ) || exit;

define( 'MRLS_VERSION', '1.0.0' );
define( 'MRLS_FILE',    __FILE__ );
define( 'MRLS_DIR',     plugin_dir_path( __FILE__ ) );
define( 'MRLS_URL',     plugin_dir_url( __FILE__ ) );

spl_autoload_register( function ( string $class ) {
	$prefix = 'MyRock\\LicenseServer\\';
	if ( strncmp( $prefix, $class, strlen( $prefix ) ) !== 0 ) {
		return;
	}
	$relative = substr( $class, strlen( $prefix ) );
	$file     = MRLS_DIR . 'includes/' . str_replace( '\\', '/', $relative ) . '.php';
	if ( file_exists( $file ) ) {
		require $file;
	}
} );

register_activation_hook( __FILE__, [ 'MyRock\\LicenseServer\\Activator', 'activate' ] );
add_action( 'plugins_loaded', [ 'MyRock\\LicenseServer\\Plugin', 'get_instance' ] );

<?php

namespace MyRock\LicenseServer;

defined( 'ABSPATH' ) || exit;

class Plugin {

	private static ?self $instance = null;

	public static function get_instance(): self {
		if ( null === self::$instance ) {
			self::$instance = new self();
		}
		return self::$instance;
	}

	private function __construct() {
		add_action( 'rest_api_init', [ new Api\RestApi(), 'register_routes' ] );
		if ( is_admin() ) {
			( new Admin\Admin() )->init();
		}
	}
}

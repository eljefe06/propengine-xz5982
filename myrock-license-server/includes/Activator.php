<?php

namespace MyRock\LicenseServer;

defined( 'ABSPATH' ) || exit;

class Activator {

	public static function activate(): void {
		Database\Schema::create_tables();
	}
}

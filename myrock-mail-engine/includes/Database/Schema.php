<?php
/**
 * Database schema definition and table creation.
 * Uses dbDelta() for safe upgrades.
 *
 * @package MyRock\MailEngine\Database
 */

namespace MyRock\MailEngine\Database;

defined( 'ABSPATH' ) || exit;

class Schema {

	/**
	 * Create or upgrade all plugin tables.
	 */
	public static function create_tables(): void {
		global $wpdb;
		require_once ABSPATH . 'wp-admin/includes/upgrade.php';

		$charset = $wpdb->get_charset_collate();
		$p       = $wpdb->prefix;

		// ── Contacts ──────────────────────────────────────────────────────
		dbDelta( "CREATE TABLE `{$p}mrme_contacts` (
			`id`          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
			`email`       VARCHAR(255)    NOT NULL,
			`first_name`  VARCHAR(100)    NOT NULL DEFAULT '',
			`last_name`   VARCHAR(100)    NOT NULL DEFAULT '',
			`phone`       VARCHAR(50)     NOT NULL DEFAULT '',
			`company`     VARCHAR(150)    NOT NULL DEFAULT '',
			`status`      ENUM('subscribed','unsubscribed','pending','bounced','complained') NOT NULL DEFAULT 'subscribed',
			`source`      VARCHAR(100)    NOT NULL DEFAULT 'manual',
			`ip_address`  VARCHAR(45)     NOT NULL DEFAULT '',
			`notes`       TEXT,
			`meta`        LONGTEXT,
			`created_at`  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
			`updated_at`  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
			PRIMARY KEY (`id`),
			UNIQUE KEY `email` (`email`),
			KEY `status` (`status`),
			KEY `created_at` (`created_at`)
		) {$charset};" );

		// ── Lists ─────────────────────────────────────────────────────────
		dbDelta( "CREATE TABLE `{$p}mrme_lists` (
			`id`          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
			`name`        VARCHAR(150)    NOT NULL,
			`slug`        VARCHAR(150)    NOT NULL,
			`description` TEXT,
			`is_public`   TINYINT(1)      NOT NULL DEFAULT 1,
			`created_at`  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
			PRIMARY KEY (`id`),
			UNIQUE KEY `slug` (`slug`)
		) {$charset};" );

		// ── Contact ↔ Lists pivot ─────────────────────────────────────────
		dbDelta( "CREATE TABLE `{$p}mrme_contact_lists` (
			`contact_id`  BIGINT UNSIGNED NOT NULL,
			`list_id`     BIGINT UNSIGNED NOT NULL,
			`subscribed_at` DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
			PRIMARY KEY (`contact_id`, `list_id`),
			KEY `list_id` (`list_id`)
		) {$charset};" );

		// ── Tags ──────────────────────────────────────────────────────────
		dbDelta( "CREATE TABLE `{$p}mrme_tags` (
			`id`    BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
			`name`  VARCHAR(100)    NOT NULL,
			`slug`  VARCHAR(100)    NOT NULL,
			`color` VARCHAR(7)      NOT NULL DEFAULT '#BFA26A',
			PRIMARY KEY (`id`),
			UNIQUE KEY `slug` (`slug`)
		) {$charset};" );

		// ── Contact ↔ Tags pivot ──────────────────────────────────────────
		dbDelta( "CREATE TABLE `{$p}mrme_contact_tags` (
			`contact_id` BIGINT UNSIGNED NOT NULL,
			`tag_id`     BIGINT UNSIGNED NOT NULL,
			`tagged_at`  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
			PRIMARY KEY (`contact_id`, `tag_id`),
			KEY `tag_id` (`tag_id`)
		) {$charset};" );

		// ── Campaigns ─────────────────────────────────────────────────────
		dbDelta( "CREATE TABLE `{$p}mrme_campaigns` (
			`id`            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
			`title`         VARCHAR(255)    NOT NULL,
			`subject`       VARCHAR(255)    NOT NULL DEFAULT '',
			`preheader`     VARCHAR(255)    NOT NULL DEFAULT '',
			`from_name`     VARCHAR(150)    NOT NULL DEFAULT '',
			`from_email`    VARCHAR(255)    NOT NULL DEFAULT '',
			`reply_to`      VARCHAR(255)    NOT NULL DEFAULT '',
			`content_html`  LONGTEXT,
			`content_text`  LONGTEXT,
			`status`        ENUM('draft','scheduled','sending','sent','cancelled') NOT NULL DEFAULT 'draft',
			`list_ids`      TEXT,
			`tag_ids`       TEXT,
			`scheduled_at`  DATETIME,
			`sent_at`       DATETIME,
			`total_sent`    INT UNSIGNED    NOT NULL DEFAULT 0,
			`total_opens`   INT UNSIGNED    NOT NULL DEFAULT 0,
			`total_clicks`  INT UNSIGNED    NOT NULL DEFAULT 0,
			`total_bounces` INT UNSIGNED    NOT NULL DEFAULT 0,
			`total_unsubs`  INT UNSIGNED    NOT NULL DEFAULT 0,
			`created_at`    DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
			`updated_at`    DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
			PRIMARY KEY (`id`),
			KEY `status` (`status`),
			KEY `scheduled_at` (`scheduled_at`)
		) {$charset};" );

		// ── Send Logs (one row per contact per campaign) ──────────────────
		dbDelta( "CREATE TABLE `{$p}mrme_send_logs` (
			`id`          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
			`campaign_id` BIGINT UNSIGNED NOT NULL,
			`contact_id`  BIGINT UNSIGNED NOT NULL,
			`email`       VARCHAR(255)    NOT NULL,
			`status`      ENUM('pending','sent','failed','bounced') NOT NULL DEFAULT 'pending',
			`sent_at`     DATETIME,
			`error`       TEXT,
			PRIMARY KEY (`id`),
			KEY `campaign_id` (`campaign_id`),
			KEY `contact_id`  (`contact_id`),
			KEY `status`      (`status`)
		) {$charset};" );

		// ── Events (opens, clicks, unsubscribes) ──────────────────────────
		dbDelta( "CREATE TABLE `{$p}mrme_events` (
			`id`          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
			`campaign_id` BIGINT UNSIGNED NOT NULL,
			`contact_id`  BIGINT UNSIGNED NOT NULL,
			`type`        ENUM('open','click','unsubscribe','bounce','complaint') NOT NULL,
			`data`        TEXT,
			`ip_address`  VARCHAR(45)     NOT NULL DEFAULT '',
			`user_agent`  VARCHAR(255)    NOT NULL DEFAULT '',
			`created_at`  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
			PRIMARY KEY (`id`),
			KEY `campaign_id` (`campaign_id`),
			KEY `contact_id`  (`contact_id`),
			KEY `type`        (`type`)
		) {$charset};" );

		// ── Forms ─────────────────────────────────────────────────────────
		dbDelta( "CREATE TABLE `{$p}mrme_forms` (
			`id`              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
			`name`            VARCHAR(150)    NOT NULL,
			`fields`          LONGTEXT,
			`list_ids`        TEXT,
			`tag_ids`         TEXT,
			`double_optin`    TINYINT(1)      NOT NULL DEFAULT 0,
			`success_message` TEXT,
			`redirect_url`    VARCHAR(500)    NOT NULL DEFAULT '',
			`status`          ENUM('active','inactive') NOT NULL DEFAULT 'active',
			`submissions`     INT UNSIGNED    NOT NULL DEFAULT 0,
			`created_at`      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
			PRIMARY KEY (`id`)
		) {$charset};" );

		// ── Automations ───────────────────────────────────────────────────
		dbDelta( "CREATE TABLE `{$p}mrme_automations` (
			`id`          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
			`name`        VARCHAR(150)    NOT NULL,
			`trigger`     VARCHAR(50)     NOT NULL DEFAULT 'subscribe',
			`trigger_data`TEXT,
			`status`      ENUM('active','inactive') NOT NULL DEFAULT 'active',
			`created_at`  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
			PRIMARY KEY (`id`)
		) {$charset};" );

		// ── Automation Steps ──────────────────────────────────────────────
		dbDelta( "CREATE TABLE `{$p}mrme_automation_steps` (
			`id`             BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
			`automation_id`  BIGINT UNSIGNED NOT NULL,
			`step_order`     INT UNSIGNED    NOT NULL DEFAULT 0,
			`type`           ENUM('email','wait','condition','tag','untag') NOT NULL DEFAULT 'email',
			`delay_value`    INT UNSIGNED    NOT NULL DEFAULT 0,
			`delay_unit`     ENUM('minutes','hours','days') NOT NULL DEFAULT 'days',
			`data`           LONGTEXT,
			PRIMARY KEY (`id`),
			KEY `automation_id` (`automation_id`)
		) {$charset};" );

		// ── Automation Queue ──────────────────────────────────────────────
		dbDelta( "CREATE TABLE `{$p}mrme_automation_queue` (
			`id`             BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
			`automation_id`  BIGINT UNSIGNED NOT NULL,
			`step_id`        BIGINT UNSIGNED NOT NULL,
			`contact_id`     BIGINT UNSIGNED NOT NULL,
			`run_at`         DATETIME        NOT NULL,
			`status`         ENUM('pending','done','failed') NOT NULL DEFAULT 'pending',
			`created_at`     DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
			PRIMARY KEY (`id`),
			KEY `run_at`     (`run_at`),
			KEY `contact_id` (`contact_id`)
		) {$charset};" );
	}

	/**
	 * Seed default list and automation on first activation.
	 */
	public static function seed_defaults(): void {
		global $wpdb;
		$p = $wpdb->prefix;

		// Default list
		$existing = $wpdb->get_var( "SELECT COUNT(*) FROM `{$p}mrme_lists`" );
		if ( (int) $existing === 0 ) {
			$wpdb->insert( "{$p}mrme_lists", [
				'name'        => 'Lista principal',
				'slug'        => 'lista-principal',
				'description' => 'Lista general de suscriptores.',
				'is_public'   => 1,
			] );
		}

		// Default form
		$existing_forms = $wpdb->get_var( "SELECT COUNT(*) FROM `{$p}mrme_forms`" );
		if ( (int) $existing_forms === 0 ) {
			$default_fields = json_encode( [
				[ 'name' => 'first_name', 'label' => 'Nombre',         'type' => 'text',  'required' => true  ],
				[ 'name' => 'email',      'label' => 'Correo electrónico','type' => 'email', 'required' => true  ],
			] );
			$wpdb->insert( "{$p}mrme_forms", [
				'name'            => 'Formulario de suscripción',
				'fields'          => $default_fields,
				'list_ids'        => '1',
				'double_optin'    => 0,
				'success_message' => '¡Gracias! Te has suscrito correctamente.',
				'status'          => 'active',
			] );
		}

		// Default welcome automation
		$existing_auto = $wpdb->get_var( "SELECT COUNT(*) FROM `{$p}mrme_automations`" );
		if ( (int) $existing_auto === 0 ) {
			$settings = get_option( 'mrme_settings', [] );
			$from_name  = $settings['from_name']  ?? get_bloginfo( 'name' );
			$from_email = $settings['from_email'] ?? get_option( 'admin_email' );

			$wpdb->insert( "{$p}mrme_automations", [
				'name'    => 'Bienvenida al newsletter',
				'trigger' => 'subscribe',
				'status'  => 'active',
			] );
			$auto_id = $wpdb->insert_id;

			$welcome_html  = '<h1>Bienvenido/a, {{first_name}}.</h1>';
			$welcome_html .= '<p>Gracias por suscribirte al newsletter de ' . esc_html( $from_name ) . '.</p>';
			$welcome_html .= '<p>Aquí encontrarás artículos sobre diseño, desarrollo y estrategia digital.</p>';
			$welcome_html .= '<p>— El equipo de ' . esc_html( $from_name ) . '</p>';
			$welcome_html .= '<p style="font-size:12px;color:#999;"><a href="{{unsubscribe_url}}">Cancelar suscripción</a></p>';

			$wpdb->insert( "{$p}mrme_automation_steps", [
				'automation_id' => $auto_id,
				'step_order'    => 1,
				'type'          => 'email',
				'delay_value'   => 0,
				'delay_unit'    => 'minutes',
				'data'          => json_encode( [
					'subject'      => '¡Bienvenido/a a ' . $from_name . '!',
					'content_html' => $welcome_html,
					'from_name'    => $from_name,
					'from_email'   => $from_email,
				] ),
			] );
		}
	}
}

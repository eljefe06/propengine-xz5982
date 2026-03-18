<?php

namespace MyRock\MailEngine\Mail;

defined( 'ABSPATH' ) || exit;

use MyRock\MailEngine\Mail\Providers\ProviderInterface;
use MyRock\MailEngine\Mail\Providers\WpMailProvider;
use MyRock\MailEngine\Mail\Providers\SmtpProvider;
use MyRock\MailEngine\Mail\Providers\MailgunProvider;

/**
 * Class MailManager
 *
 * Central static facade for the MyRock Mail Engine mailing system.
 *
 * Responsibilities:
 *  - Resolve and cache the active mail provider based on plugin settings.
 *  - Provide a simple send() API that accepts a MailMessage DTO.
 *  - Provide a send_raw() convenience helper for one-liner sends.
 *  - Expose the configured default From address/name.
 *
 * Usage:
 *
 *   // Fluent MailMessage API:
 *   $result = MailManager::send(
 *       ( new MailMessage() )
 *           ->setTo( 'user@example.com', 'Jane' )
 *           ->setSubject( 'Welcome!' )
 *           ->setHtml( '<p>Welcome aboard!</p>' )
 *           ->setText( 'Welcome aboard!' )
 *   );
 *
 *   // One-liner helper:
 *   MailManager::send_raw( 'user@example.com', 'Hello', '<p>Hello</p>' );
 *
 * @package MyRock\MailEngine\Mail
 */
class MailManager {

	/**
	 * Cached provider instance.
	 *
	 * Populated on the first call to get_provider() and reused for the
	 * remainder of the request.
	 *
	 * @var ProviderInterface|null
	 */
	private static ?ProviderInterface $provider = null;

	// Prevent direct instantiation — all methods are static.
	private function __construct() {}

	// -------------------------------------------------------------------------
	// Provider resolution
	// -------------------------------------------------------------------------

	/**
	 * Return the active mail provider instance.
	 *
	 * The provider is determined by the `mail_provider` key inside the
	 * `mrme_settings` WordPress option.  Supported values:
	 *
	 *   - 'wp_mail' (default) — delegates to WpMailProvider.
	 *   - 'smtp'              — delegates to SmtpProvider.
	 *
	 * The resolved instance is cached for the lifetime of the request so that
	 * provider objects are not re-created on every send call.
	 *
	 * @return ProviderInterface
	 */
	public static function get_provider(): ProviderInterface {
		if ( self::$provider instanceof ProviderInterface ) {
			return self::$provider;
		}

		$settings      = get_option( 'mrme_settings', [] );
		$provider_key  = isset( $settings['mail_provider'] )
			? (string) $settings['mail_provider']
			: 'wp_mail';

		switch ( $provider_key ) {
			case 'smtp':
				self::$provider = new SmtpProvider();
				break;

			case 'mailgun':
				self::$provider = new MailgunProvider();
				break;

			case 'wp_mail':
			default:
				self::$provider = new WpMailProvider();
				break;
		}

		return self::$provider;
	}

	/**
	 * Replace the cached provider with a custom instance.
	 *
	 * This is primarily intended for unit-testing so a mock provider can be
	 * injected without touching WordPress options.
	 *
	 * @param ProviderInterface $provider Provider instance to use.
	 *
	 * @return void
	 */
	public static function set_provider( ProviderInterface $provider ): void {
		self::$provider = $provider;
	}

	/**
	 * Clear the cached provider, forcing re-resolution on the next send call.
	 *
	 * Useful when settings change at runtime (e.g. in admin save handlers).
	 *
	 * @return void
	 */
	public static function reset_provider(): void {
		self::$provider = null;
	}

	// -------------------------------------------------------------------------
	// Send API
	// -------------------------------------------------------------------------

	/**
	 * Send an email using the active provider.
	 *
	 * If no From address is set on the message, the plugin's configured default
	 * is applied automatically before dispatch.
	 *
	 * On failure, the provider's last error is written to the PHP error log so
	 * that site administrators can diagnose delivery issues without exposing
	 * sensitive data to end users.
	 *
	 * @param MailMessage $message Populated MailMessage instance.
	 *
	 * @return bool True when the provider accepted the message, false otherwise.
	 */
	public static function send( MailMessage $message ): bool {
		$provider = self::get_provider();

		$data = $message->toArray();

		// Apply default From when the caller has not supplied one ---------------
		if ( empty( $data['from_email'] ) ) {
			$default = self::get_default_from();
			$message->setFrom( $default['email'], $default['name'] );
			$data = $message->toArray();
		}

		$result = $provider->send( $data );

		if ( ! $result ) {
			$error = $provider->get_last_error();
			$log   = sprintf(
				'[MyRock Mail Engine] Failed to send email to "%s" (subject: "%s"). Provider error: %s',
				$data['to'] ?? '',
				$data['subject'] ?? '',
				! empty( $error ) ? $error : '(no error detail available)'
			);
			error_log( $log ); // phpcs:ignore WordPress.PHP.DevelopmentFunctions.error_log_error_log
		}

		return $result;
	}

	/**
	 * Build and send an email without manually constructing a MailMessage.
	 *
	 * This helper covers the common case where only the core fields are needed.
	 * For more advanced scenarios (custom headers, plain-text AltBody, etc.) use
	 * the MailMessage builder together with send().
	 *
	 * @param string $to         Recipient email address.
	 * @param string $subject    Email subject line.
	 * @param string $html       HTML body of the email.
	 * @param string $from_email Sender email address.  Uses plugin default when empty.
	 * @param string $from_name  Sender display name.  Uses plugin default when empty.
	 * @param string $reply_to   Reply-To email address (optional).
	 *
	 * @return bool True on success, false on failure.
	 */
	public static function send_raw(
		string $to,
		string $subject,
		string $html,
		string $from_email = '',
		string $from_name  = '',
		string $reply_to   = ''
	): bool {
		// Resolve defaults for From when not explicitly supplied ----------------
		if ( empty( $from_email ) || empty( $from_name ) ) {
			$default = self::get_default_from();

			if ( empty( $from_email ) ) {
				$from_email = $default['email'];
			}

			if ( empty( $from_name ) ) {
				$from_name = $default['name'];
			}
		}

		$message = ( new MailMessage() )
			->setTo( $to )
			->setSubject( $subject )
			->setHtml( $html )
			->setFrom( $from_email, $from_name );

		if ( ! empty( $reply_to ) ) {
			$message->setReplyTo( $reply_to );
		}

		return self::send( $message );
	}

	// -------------------------------------------------------------------------
	// Default From
	// -------------------------------------------------------------------------

	/**
	 * Return the default From email address and display name for outgoing mail.
	 *
	 * Resolution order for the email address:
	 *  1. mrme_settings['from_email'] (plugin setting).
	 *  2. WordPress admin email via get_bloginfo('admin_email').
	 *
	 * Resolution order for the display name:
	 *  1. mrme_settings['from_name'] (plugin setting).
	 *  2. Site name via get_bloginfo('name').
	 *
	 * @return array {
	 *     @type string $email Default sender email address.
	 *     @type string $name  Default sender display name.
	 * }
	 */
	public static function get_default_from(): array {
		$settings = get_option( 'mrme_settings', [] );

		$email = isset( $settings['from_email'] ) && ! empty( $settings['from_email'] )
			? (string) $settings['from_email']
			: (string) get_bloginfo( 'admin_email' );

		$name = isset( $settings['from_name'] ) && ! empty( $settings['from_name'] )
			? (string) $settings['from_name']
			: (string) get_bloginfo( 'name' );

		return [
			'email' => $email,
			'name'  => $name,
		];
	}
}

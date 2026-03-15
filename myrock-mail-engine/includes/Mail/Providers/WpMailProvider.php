<?php

namespace MyRock\MailEngine\Mail\Providers;

defined( 'ABSPATH' ) || exit;

/**
 * Class WpMailProvider
 *
 * Sends email using WordPress's built-in wp_mail() function.
 * Supports multipart (HTML + plain text) messages by hooking into
 * phpmailer_init to attach an AltBody before the message is dispatched.
 *
 * @package MyRock\MailEngine\Mail\Providers
 */
class WpMailProvider implements ProviderInterface {

	/**
	 * Stores the last error string after a failed send attempt.
	 *
	 * @var string
	 */
	private string $last_error = '';

	/**
	 * Plain-text body to be injected via phpmailer_init when sending multipart mail.
	 *
	 * @var string
	 */
	private string $pending_text = '';

	/**
	 * Whether we have already registered the phpmailer_init error-capture hook
	 * for the current request.
	 *
	 * @var bool
	 */
	private bool $error_hook_registered = false;

	// -------------------------------------------------------------------------
	// ProviderInterface
	// -------------------------------------------------------------------------

	/**
	 * {@inheritdoc}
	 *
	 * Builds the headers array (From, Reply-To, Content-Type) and calls
	 * wp_mail(). When a plain-text body is present, a phpmailer_init hook
	 * is used to set PHPMailer's AltBody so that a proper multipart/alternative
	 * message is sent.
	 */
	public function send( array $message ): bool {
		$this->last_error = '';

		$to      = $message['to']      ?? '';
		$subject = $message['subject'] ?? '';
		$html    = $message['html']    ?? '';
		$text    = $message['text']    ?? '';

		if ( empty( $to ) || empty( $subject ) ) {
			$this->last_error = 'WpMailProvider: "to" and "subject" are required.';
			return false;
		}

		// Build headers ---------------------------------------------------------
		$headers = $this->build_headers( $message );

		// Multipart handling ----------------------------------------------------
		if ( ! empty( $text ) ) {
			$this->pending_text = $text;
			add_action( 'phpmailer_init', [ $this, 'inject_alt_body' ], 10, 1 );
		}

		// Register error-capture hook once per instance lifetime ----------------
		if ( ! $this->error_hook_registered ) {
			add_action( 'wp_mail_failed', [ $this, 'capture_wp_mail_error' ], 10, 1 );
			$this->error_hook_registered = true;
		}

		$result = wp_mail( $to, $subject, $html, $headers );

		// Clean up multipart hook -----------------------------------------------
		if ( ! empty( $text ) ) {
			remove_action( 'phpmailer_init', [ $this, 'inject_alt_body' ], 10 );
			$this->pending_text = '';
		}

		return $result;
	}

	/**
	 * {@inheritdoc}
	 */
	public function get_last_error(): string {
		return $this->last_error;
	}

	// -------------------------------------------------------------------------
	// Internal helpers
	// -------------------------------------------------------------------------

	/**
	 * Build the headers array to be passed to wp_mail().
	 *
	 * @param array $message Message data (same shape as send()).
	 *
	 * @return string[] Indexed array of raw header strings.
	 */
	private function build_headers( array $message ): array {
		$headers = [];

		// From header -----------------------------------------------------------
		$from_email = $message['from_email'] ?? '';
		$from_name  = $message['from_name']  ?? '';

		if ( ! empty( $from_email ) ) {
			if ( ! empty( $from_name ) ) {
				$headers[] = sprintf( 'From: %s <%s>', $from_name, $from_email );
			} else {
				$headers[] = sprintf( 'From: %s', $from_email );
			}
		}

		// Reply-To header -------------------------------------------------------
		$reply_to = $message['reply_to'] ?? '';
		if ( ! empty( $reply_to ) ) {
			$headers[] = sprintf( 'Reply-To: %s', $reply_to );
		}

		// Content-Type ----------------------------------------------------------
		// Always declare HTML; AltBody handles the text part inside PHPMailer.
		$headers[] = 'Content-Type: text/html; charset=UTF-8';

		// Caller-supplied extra headers -----------------------------------------
		$extra = $message['headers'] ?? [];
		if ( is_array( $extra ) ) {
			foreach ( $extra as $header ) {
				if ( is_string( $header ) && ! empty( $header ) ) {
					$headers[] = $header;
				}
			}
		}

		return $headers;
	}

	/**
	 * phpmailer_init callback — injects the plain-text AltBody so that
	 * PHPMailer sends a multipart/alternative message.
	 *
	 * @param \PHPMailer\PHPMailer\PHPMailer $phpmailer The PHPMailer instance passed by WordPress.
	 *
	 * @return void
	 */
	public function inject_alt_body( $phpmailer ): void {
		if ( ! empty( $this->pending_text ) ) {
			$phpmailer->AltBody = $this->pending_text;
		}
	}

	/**
	 * wp_mail_failed callback — captures the WP_Error into $last_error.
	 *
	 * @param \WP_Error $error The error object passed by WordPress.
	 *
	 * @return void
	 */
	public function capture_wp_mail_error( $error ): void {
		if ( $error instanceof \WP_Error ) {
			$this->last_error = implode( ' | ', $error->get_error_messages() );
		}
	}
}

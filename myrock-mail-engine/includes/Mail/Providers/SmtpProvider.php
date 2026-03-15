<?php

namespace MyRock\MailEngine\Mail\Providers;

defined( 'ABSPATH' ) || exit;

/**
 * Class SmtpProvider
 *
 * Sends email via a custom SMTP server by overriding PHPMailer's transport
 * settings through the `phpmailer_init` hook, then delegating the actual
 * dispatch to WordPress's wp_mail().
 *
 * SMTP credentials and connection parameters are read from the
 * `mrme_settings` WordPress option:
 *
 *   - smtp_host        (string)  SMTP server hostname.
 *   - smtp_port        (int)     SMTP server port (e.g. 587, 465, 25).
 *   - smtp_encryption  (string)  'tls', 'ssl', or 'none'.
 *   - smtp_username    (string)  SMTP authentication username.
 *   - smtp_password    (string)  SMTP authentication password.
 *
 * @package MyRock\MailEngine\Mail\Providers
 */
class SmtpProvider implements ProviderInterface {

	/**
	 * Merged SMTP settings read from the mrme_settings option.
	 *
	 * @var array
	 */
	private array $settings;

	/**
	 * Last error string captured from a failed send attempt.
	 *
	 * @var string
	 */
	private string $last_error = '';

	/**
	 * Plain-text body to be injected via phpmailer_init when sending
	 * a multipart message.
	 *
	 * @var string
	 */
	private string $pending_text = '';

	/**
	 * Whether the wp_mail_failed error-capture hook has been registered.
	 *
	 * @var bool
	 */
	private bool $error_hook_registered = false;

	// -------------------------------------------------------------------------
	// Constructor
	// -------------------------------------------------------------------------

	/**
	 * SmtpProvider constructor.
	 *
	 * Reads SMTP settings from the mrme_settings WordPress option and merges
	 * them with sensible defaults so that every subsequent access is safe.
	 */
	public function __construct() {
		$stored = get_option( 'mrme_settings', [] );

		$this->settings = wp_parse_args(
			is_array( $stored ) ? $stored : [],
			[
				'smtp_host'       => '',
				'smtp_port'       => 587,
				'smtp_encryption' => 'tls',
				'smtp_username'   => '',
				'smtp_password'   => '',
			]
		);
	}

	// -------------------------------------------------------------------------
	// ProviderInterface
	// -------------------------------------------------------------------------

	/**
	 * {@inheritdoc}
	 *
	 * Registers the phpmailer_init hook to inject SMTP settings, calls
	 * wp_mail(), then removes the hook to avoid affecting any subsequent
	 * wp_mail() calls made by other code in the same request.
	 */
	public function send( array $message ): bool {
		$this->last_error = '';

		$to      = $message['to']      ?? '';
		$subject = $message['subject'] ?? '';
		$html    = $message['html']    ?? '';
		$text    = $message['text']    ?? '';

		if ( empty( $to ) || empty( $subject ) ) {
			$this->last_error = 'SmtpProvider: "to" and "subject" are required.';
			return false;
		}

		// Register error-capture hook once per instance lifetime ----------------
		if ( ! $this->error_hook_registered ) {
			add_action( 'wp_mail_failed', [ $this, 'capture_wp_mail_error' ], 10, 1 );
			$this->error_hook_registered = true;
		}

		// Multipart handling ----------------------------------------------------
		if ( ! empty( $text ) ) {
			$this->pending_text = $text;
		}

		// Hook into phpmailer_init to configure SMTP and inject AltBody ---------
		add_action( 'phpmailer_init', [ $this, 'configure_phpmailer' ], 10, 1 );

		$headers = $this->build_headers( $message );

		$result = wp_mail( $to, $subject, $html, $headers );

		// Clean up ---------------------------------------------------------------
		remove_action( 'phpmailer_init', [ $this, 'configure_phpmailer' ], 10 );
		$this->pending_text = '';

		return $result;
	}

	/**
	 * {@inheritdoc}
	 */
	public function get_last_error(): string {
		return $this->last_error;
	}

	// -------------------------------------------------------------------------
	// phpmailer_init callback
	// -------------------------------------------------------------------------

	/**
	 * Configure the PHPMailer instance to use the plugin's SMTP settings.
	 *
	 * This method is registered as a `phpmailer_init` action callback for the
	 * duration of each send() call only, then removed immediately afterwards.
	 *
	 * @param \PHPMailer\PHPMailer\PHPMailer $phpmailer The PHPMailer instance
	 *                                                  provided by WordPress.
	 *
	 * @return void
	 */
	public function configure_phpmailer( $phpmailer ): void {
		// Switch transport to SMTP ----------------------------------------------
		$phpmailer->isSMTP();

		// Connection parameters -------------------------------------------------
		$phpmailer->Host = $this->settings['smtp_host'];
		$phpmailer->Port = (int) $this->settings['smtp_port'];

		// Encryption ------------------------------------------------------------
		$encryption = strtolower( (string) $this->settings['smtp_encryption'] );

		switch ( $encryption ) {
			case 'ssl':
				$phpmailer->SMTPSecure = \PHPMailer\PHPMailer\PHPMailer::ENCRYPTION_SMTPS;
				break;

			case 'tls':
				$phpmailer->SMTPSecure = \PHPMailer\PHPMailer\PHPMailer::ENCRYPTION_STARTTLS;
				break;

			case 'none':
			default:
				$phpmailer->SMTPSecure = '';
				$phpmailer->SMTPAutoTLS = false;
				break;
		}

		// Authentication --------------------------------------------------------
		$username = (string) $this->settings['smtp_username'];
		$password = (string) $this->settings['smtp_password'];

		if ( ! empty( $username ) ) {
			$phpmailer->SMTPAuth = true;
			$phpmailer->Username = $username;
			$phpmailer->Password = $password;
		} else {
			$phpmailer->SMTPAuth = false;
		}

		// Multipart AltBody -----------------------------------------------------
		if ( ! empty( $this->pending_text ) ) {
			$phpmailer->AltBody = $this->pending_text;
		}
	}

	// -------------------------------------------------------------------------
	// Internal helpers
	// -------------------------------------------------------------------------

	/**
	 * Build the headers array to pass to wp_mail().
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

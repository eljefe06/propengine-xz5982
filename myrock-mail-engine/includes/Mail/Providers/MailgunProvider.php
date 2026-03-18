<?php

namespace MyRock\MailEngine\Mail\Providers;

defined( 'ABSPATH' ) || exit;

/**
 * Class MailgunProvider
 *
 * Sends email via the Mailgun HTTP API using wp_remote_post().
 * No external library required — only the API key and domain.
 *
 * Settings read from mrme_settings WordPress option:
 *   - mailgun_api_key    (string)  Private API key from Mailgun dashboard.
 *   - mailgun_domain     (string)  Sending domain registered in Mailgun (e.g. mg.mysite.com).
 *   - mailgun_region     (string)  'us' (default) or 'eu' (European data center).
 *
 * @package MyRock\MailEngine\Mail\Providers
 */
class MailgunProvider implements ProviderInterface {

	/** Mailgun US API endpoint. */
	const ENDPOINT_US = 'https://api.mailgun.net/v3';

	/** Mailgun EU API endpoint. */
	const ENDPOINT_EU = 'https://api.eu.mailgun.net/v3';

	/** @var array Plugin settings. */
	private array $settings;

	/** @var string Last error message. */
	private string $last_error = '';

	/**
	 * Constructor — reads and merges Mailgun settings.
	 */
	public function __construct() {
		$stored = get_option( 'mrme_settings', [] );

		$this->settings = wp_parse_args(
			is_array( $stored ) ? $stored : [],
			[
				'mailgun_api_key' => '',
				'mailgun_domain'  => '',
				'mailgun_region'  => 'us',
			]
		);
	}

	// -------------------------------------------------------------------------
	// ProviderInterface
	// -------------------------------------------------------------------------

	/**
	 * Send an email via the Mailgun API.
	 *
	 * @param array $message {
	 *   @type string       $to           Recipient email address.
	 *   @type string       $to_name      Recipient display name (optional).
	 *   @type string       $subject      Email subject.
	 *   @type string       $html         HTML body.
	 *   @type string       $text         Plain-text body (optional).
	 *   @type string       $from_email   Sender email.
	 *   @type string       $from_name    Sender display name.
	 *   @type string       $reply_to     Reply-To address (optional).
	 *   @type string[]     $headers      Extra headers (e.g. List-Unsubscribe).
	 * }
	 * @return bool
	 */
	public function send( array $message ): bool {
		$this->last_error = '';

		$api_key = trim( (string) ( $this->settings['mailgun_api_key'] ?? '' ) );
		$domain  = trim( (string) ( $this->settings['mailgun_domain']  ?? '' ) );

		if ( empty( $api_key ) || empty( $domain ) ) {
			$this->last_error = __( 'Mailgun API key or domain is not configured.', 'myrock-mail-engine' );
			return false;
		}

		$to      = $message['to']      ?? '';
		$subject = $message['subject'] ?? '';
		$html    = $message['html']    ?? '';
		$text    = $message['text']    ?? '';

		if ( empty( $to ) || empty( $subject ) ) {
			$this->last_error = __( '"to" and "subject" are required fields.', 'myrock-mail-engine' );
			return false;
		}

		// Build From header.
		$from_email = $message['from_email'] ?? '';
		$from_name  = $message['from_name']  ?? '';
		$from       = ! empty( $from_name )
			? sprintf( '%s <%s>', $from_name, $from_email )
			: $from_email;

		// Build To header.
		$to_name   = $message['to_name'] ?? '';
		$to_header = ! empty( $to_name )
			? sprintf( '%s <%s>', $to_name, $to )
			: $to;

		// Compose POST body.
		$body = [
			'from'    => $from,
			'to'      => $to_header,
			'subject' => $subject,
			'html'    => $html,
		];

		if ( ! empty( $text ) ) {
			$body['text'] = $text;
		}

		// Reply-To.
		$reply_to = $message['reply_to'] ?? '';
		if ( ! empty( $reply_to ) ) {
			$body['h:Reply-To'] = $reply_to;
		}

		// Extra headers (e.g. List-Unsubscribe).
		$extra_headers = $message['headers'] ?? [];
		if ( is_array( $extra_headers ) ) {
			foreach ( $extra_headers as $header ) {
				if ( is_string( $header ) && strpos( $header, ':' ) !== false ) {
					[ $hname, $hval ] = array_map( 'trim', explode( ':', $header, 2 ) );
					$body[ 'h:' . $hname ] = $hval;
				}
			}
		}

		// API endpoint.
		$region   = strtolower( (string) ( $this->settings['mailgun_region'] ?? 'us' ) );
		$base_url = ( 'eu' === $region ) ? self::ENDPOINT_EU : self::ENDPOINT_US;
		$endpoint = $base_url . '/' . $domain . '/messages';

		// HTTP request.
		$response = wp_remote_post( $endpoint, [
			'timeout'  => 20,
			'headers'  => [
				'Authorization' => 'Basic ' . base64_encode( 'api:' . $api_key ),
			],
			'body'     => $body,
		] );

		if ( is_wp_error( $response ) ) {
			$this->last_error = $response->get_error_message();
			return false;
		}

		$code = wp_remote_retrieve_response_code( $response );
		if ( $code < 200 || $code >= 300 ) {
			$raw_body = wp_remote_retrieve_body( $response );
			$decoded  = json_decode( $raw_body, true );
			$this->last_error = isset( $decoded['message'] )
				? (string) $decoded['message']
				: sprintf( 'Mailgun HTTP %d: %s', $code, $raw_body );
			return false;
		}

		return true;
	}

	/**
	 * {@inheritdoc}
	 */
	public function get_last_error(): string {
		return $this->last_error;
	}
}

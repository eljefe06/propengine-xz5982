<?php

namespace MyRock\MailEngine\Mail\Providers;

defined( 'ABSPATH' ) || exit;

/**
 * Interface ProviderInterface
 *
 * All mail providers must implement this interface.
 *
 * @package MyRock\MailEngine\Mail\Providers
 */
interface ProviderInterface {

	/**
	 * Send an email message.
	 *
	 * @param array $message {
	 *     Mail message data.
	 *
	 *     @type string $to         Recipient email address.
	 *     @type string $to_name    Recipient display name.
	 *     @type string $subject    Email subject line.
	 *     @type string $html       HTML body of the email.
	 *     @type string $text       Plain-text body of the email (optional).
	 *     @type string $from_email Sender email address.
	 *     @type string $from_name  Sender display name.
	 *     @type string $reply_to   Reply-To email address (optional).
	 *     @type array  $headers    Additional raw headers (optional).
	 * }
	 *
	 * @return bool True on success, false on failure.
	 */
	public function send( array $message ): bool;

	/**
	 * Return the last error message produced by a failed send attempt.
	 *
	 * @return string Empty string when no error has occurred.
	 */
	public function get_last_error(): string;
}

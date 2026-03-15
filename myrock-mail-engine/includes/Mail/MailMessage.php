<?php

namespace MyRock\MailEngine\Mail;

defined( 'ABSPATH' ) || exit;

/**
 * Class MailMessage
 *
 * Immutable-style data-transfer object and fluent builder for email messages.
 * Each setter returns $this so calls can be chained:
 *
 *   $msg = ( new MailMessage() )
 *       ->setTo( 'user@example.com', 'Jane Doe' )
 *       ->setSubject( 'Hello!' )
 *       ->setHtml( '<p>Hello</p>' )
 *       ->setText( 'Hello' )
 *       ->setFrom( 'noreply@example.com', 'My Site' );
 *
 * @package MyRock\MailEngine\Mail
 */
class MailMessage {

	// -------------------------------------------------------------------------
	// Properties
	// -------------------------------------------------------------------------

	/**
	 * Recipient email address.
	 *
	 * @var string
	 */
	private string $to = '';

	/**
	 * Recipient display name.
	 *
	 * @var string
	 */
	private string $to_name = '';

	/**
	 * Email subject line.
	 *
	 * @var string
	 */
	private string $subject = '';

	/**
	 * HTML body of the email.
	 *
	 * @var string
	 */
	private string $html = '';

	/**
	 * Plain-text body of the email (used as AltBody in multipart messages).
	 *
	 * @var string
	 */
	private string $text = '';

	/**
	 * Sender email address.
	 *
	 * @var string
	 */
	private string $from_email = '';

	/**
	 * Sender display name.
	 *
	 * @var string
	 */
	private string $from_name = '';

	/**
	 * Reply-To email address.
	 *
	 * @var string
	 */
	private string $reply_to = '';

	/**
	 * Additional raw email headers.
	 *
	 * @var string[]
	 */
	private array $headers = [];

	// -------------------------------------------------------------------------
	// Builder methods
	// -------------------------------------------------------------------------

	/**
	 * Set the recipient address and optional display name.
	 *
	 * @param string $email Recipient email address.
	 * @param string $name  Recipient display name (optional).
	 *
	 * @return self
	 */
	public function setTo( string $email, string $name = '' ): self {
		$this->to      = $email;
		$this->to_name = $name;
		return $this;
	}

	/**
	 * Set the email subject line.
	 *
	 * @param string $subject Subject text.
	 *
	 * @return self
	 */
	public function setSubject( string $subject ): self {
		$this->subject = $subject;
		return $this;
	}

	/**
	 * Set the HTML body of the email.
	 *
	 * @param string $html HTML content.
	 *
	 * @return self
	 */
	public function setHtml( string $html ): self {
		$this->html = $html;
		return $this;
	}

	/**
	 * Set the plain-text body of the email.
	 *
	 * When provided, the message will be sent as multipart/alternative so that
	 * mail clients that cannot render HTML still receive readable content.
	 *
	 * @param string $text Plain-text content.
	 *
	 * @return self
	 */
	public function setText( string $text ): self {
		$this->text = $text;
		return $this;
	}

	/**
	 * Set the sender address and optional display name.
	 *
	 * @param string $email Sender email address.
	 * @param string $name  Sender display name (optional).
	 *
	 * @return self
	 */
	public function setFrom( string $email, string $name = '' ): self {
		$this->from_email = $email;
		$this->from_name  = $name;
		return $this;
	}

	/**
	 * Set the Reply-To email address.
	 *
	 * @param string $email Reply-To email address.
	 *
	 * @return self
	 */
	public function setReplyTo( string $email ): self {
		$this->reply_to = $email;
		return $this;
	}

	/**
	 * Append a raw header string to the headers list.
	 *
	 * Each call appends one header; e.g.:
	 *   $msg->addHeader( 'X-Custom-Header: value' );
	 *
	 * @param string $header Raw header string (e.g. "X-Foo: bar").
	 *
	 * @return self
	 */
	public function addHeader( string $header ): self {
		$this->headers[] = $header;
		return $this;
	}

	// -------------------------------------------------------------------------
	// Output
	// -------------------------------------------------------------------------

	/**
	 * Export all message data as an associative array suitable for passing
	 * directly to ProviderInterface::send().
	 *
	 * @return array {
	 *     @type string   $to         Recipient email.
	 *     @type string   $to_name    Recipient name.
	 *     @type string   $subject    Subject line.
	 *     @type string   $html       HTML body.
	 *     @type string   $text       Plain-text body.
	 *     @type string   $from_email Sender email.
	 *     @type string   $from_name  Sender name.
	 *     @type string   $reply_to   Reply-To email.
	 *     @type string[] $headers    Extra raw headers.
	 * }
	 */
	public function toArray(): array {
		return [
			'to'         => $this->to,
			'to_name'    => $this->to_name,
			'subject'    => $this->subject,
			'html'       => $this->html,
			'text'       => $this->text,
			'from_email' => $this->from_email,
			'from_name'  => $this->from_name,
			'reply_to'   => $this->reply_to,
			'headers'    => $this->headers,
		];
	}
}

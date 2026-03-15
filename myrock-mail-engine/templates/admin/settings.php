<?php
/**
 * Admin template: Plugin Settings.
 *
 * Renders the full settings page. All option values are retrieved via get_option()
 * directly in the template. The SettingsPage controller calls settings_fields()
 * and does_settings_sections() if using the Settings API, but we handle the form
 * manually here for maximum control.
 *
 * @package MyRock\MailEngine
 */

defined( 'ABSPATH' ) || exit;

// Read current option values.
$from_name      = get_option( 'mrme_from_name',        get_bloginfo( 'name' ) );
$from_email     = get_option( 'mrme_from_email',       get_option( 'admin_email' ) );
$reply_to       = get_option( 'mrme_reply_to',         $from_email );
$mail_provider  = get_option( 'mrme_mail_provider',    'wp_mail' );
$smtp_host      = get_option( 'mrme_smtp_host',        '' );
$smtp_port      = get_option( 'mrme_smtp_port',        '587' );
$smtp_enc       = get_option( 'mrme_smtp_encryption',  'tls' );
$smtp_user      = get_option( 'mrme_smtp_username',    '' );
$smtp_pass      = get_option( 'mrme_smtp_password',    '' );
$delete_on_uninstall = (bool) get_option( 'mrme_delete_on_uninstall', false );
$api_key        = get_option( 'mrme_api_key',          '' );

// Notice from redirect.
// phpcs:disable WordPress.Security.NonceVerification.Recommended
$saved = isset( $_GET['mrme_settings_saved'] ) && '1' === $_GET['mrme_settings_saved'];
// phpcs:enable
?>
<div class="wrap mrme-wrap">

	<h1 class="mrme-page-title"><?php esc_html_e( 'Settings', 'myrock-mail-engine' ); ?></h1>

	<?php if ( $saved ) : ?>
	<div class="notice notice-success is-dismissible">
		<p><?php esc_html_e( 'Settings saved successfully.', 'myrock-mail-engine' ); ?></p>
	</div>
	<?php endif; ?>

	<form
		method="POST"
		action="<?php echo esc_url( admin_url( 'admin-post.php' ) ); ?>"
		class="mrme-settings-form"
		novalidate
	>
		<?php wp_nonce_field( 'mrme_save_settings', '_wpnonce' ); ?>
		<input type="hidden" name="action" value="mrme_save_settings">

		<!-- ============================================================== -->
		<!-- Section 1: Sender Identity                                      -->
		<!-- ============================================================== -->
		<div class="mrme-form-section mrme-form-section--card">
			<h2 class="mrme-form-section__title"><?php esc_html_e( 'Sender Identity', 'myrock-mail-engine' ); ?></h2>
			<p class="description"><?php esc_html_e( 'Default sender details used for all campaigns and transactional emails.', 'myrock-mail-engine' ); ?></p>

			<table class="form-table mrme-settings-table">
				<tbody>
					<tr>
						<th scope="row">
							<label for="mrme-setting-from-name"><?php esc_html_e( 'From Name', 'myrock-mail-engine' ); ?></label>
						</th>
						<td>
							<input
								type="text"
								id="mrme-setting-from-name"
								name="mrme_from_name"
								class="regular-text"
								value="<?php echo esc_attr( $from_name ); ?>"
							>
							<p class="description"><?php esc_html_e( 'The name that will appear in the "From" field.', 'myrock-mail-engine' ); ?></p>
						</td>
					</tr>
					<tr>
						<th scope="row">
							<label for="mrme-setting-from-email"><?php esc_html_e( 'From Email', 'myrock-mail-engine' ); ?></label>
						</th>
						<td>
							<input
								type="email"
								id="mrme-setting-from-email"
								name="mrme_from_email"
								class="regular-text"
								value="<?php echo esc_attr( $from_email ); ?>"
								required
							>
							<p class="description"><?php esc_html_e( 'Make sure this email is authorised to send from your mail provider.', 'myrock-mail-engine' ); ?></p>
						</td>
					</tr>
					<tr>
						<th scope="row">
							<label for="mrme-setting-reply-to"><?php esc_html_e( 'Reply-To', 'myrock-mail-engine' ); ?></label>
						</th>
						<td>
							<input
								type="email"
								id="mrme-setting-reply-to"
								name="mrme_reply_to"
								class="regular-text"
								value="<?php echo esc_attr( $reply_to ); ?>"
								placeholder="<?php echo esc_attr( $from_email ); ?>"
							>
							<p class="description"><?php esc_html_e( 'Leave empty to use the From Email as reply-to.', 'myrock-mail-engine' ); ?></p>
						</td>
					</tr>
				</tbody>
			</table>
		</div>

		<!-- ============================================================== -->
		<!-- Section 2: Mail Provider                                        -->
		<!-- ============================================================== -->
		<div class="mrme-form-section mrme-form-section--card">
			<h2 class="mrme-form-section__title"><?php esc_html_e( 'Mail Provider', 'myrock-mail-engine' ); ?></h2>
			<p class="description"><?php esc_html_e( 'Choose how outgoing emails are sent.', 'myrock-mail-engine' ); ?></p>

			<table class="form-table mrme-settings-table">
				<tbody>
					<tr>
						<th scope="row"><?php esc_html_e( 'Provider', 'myrock-mail-engine' ); ?></th>
						<td>
							<fieldset>
								<legend class="screen-reader-text"><?php esc_html_e( 'Mail Provider', 'myrock-mail-engine' ); ?></legend>
								<label class="mrme-radio-label">
									<input
										type="radio"
										name="mrme_mail_provider"
										value="wp_mail"
										id="mrme-provider-wp-mail"
										<?php checked( $mail_provider, 'wp_mail' ); ?>
										class="mrme-provider-radio"
									>
									<strong><?php esc_html_e( 'WordPress wp_mail()', 'myrock-mail-engine' ); ?></strong>
									<span class="description">&nbsp;— <?php esc_html_e( 'Uses the default WordPress mail function (PHP mail or any SMTP plugin already installed).', 'myrock-mail-engine' ); ?></span>
								</label>
								<br>
								<label class="mrme-radio-label">
									<input
										type="radio"
										name="mrme_mail_provider"
										value="smtp"
										id="mrme-provider-smtp"
										<?php checked( $mail_provider, 'smtp' ); ?>
										class="mrme-provider-radio"
									>
									<strong><?php esc_html_e( 'Custom SMTP', 'myrock-mail-engine' ); ?></strong>
									<span class="description">&nbsp;— <?php esc_html_e( 'Configure a dedicated SMTP server below.', 'myrock-mail-engine' ); ?></span>
								</label>
							</fieldset>
						</td>
					</tr>
				</tbody>
			</table>
		</div>

		<!-- ============================================================== -->
		<!-- Section 3: SMTP Settings (shown only when smtp is selected)    -->
		<!-- ============================================================== -->
		<div
			class="mrme-form-section mrme-form-section--card"
			id="mrme-smtp-settings"
			<?php echo 'smtp' !== $mail_provider ? 'style="display:none;"' : ''; ?>
		>
			<h2 class="mrme-form-section__title"><?php esc_html_e( 'SMTP Settings', 'myrock-mail-engine' ); ?></h2>

			<table class="form-table mrme-settings-table">
				<tbody>
					<tr>
						<th scope="row">
							<label for="mrme-smtp-host"><?php esc_html_e( 'SMTP Host', 'myrock-mail-engine' ); ?></label>
						</th>
						<td>
							<input
								type="text"
								id="mrme-smtp-host"
								name="mrme_smtp_host"
								class="regular-text"
								value="<?php echo esc_attr( $smtp_host ); ?>"
								placeholder="smtp.example.com"
							>
						</td>
					</tr>
					<tr>
						<th scope="row">
							<label for="mrme-smtp-port"><?php esc_html_e( 'SMTP Port', 'myrock-mail-engine' ); ?></label>
						</th>
						<td>
							<input
								type="number"
								id="mrme-smtp-port"
								name="mrme_smtp_port"
								class="small-text"
								value="<?php echo esc_attr( $smtp_port ); ?>"
								min="1"
								max="65535"
							>
							<p class="description"><?php esc_html_e( 'Common ports: 25, 465 (SSL), 587 (TLS).', 'myrock-mail-engine' ); ?></p>
						</td>
					</tr>
					<tr>
						<th scope="row">
							<label for="mrme-smtp-encryption"><?php esc_html_e( 'Encryption', 'myrock-mail-engine' ); ?></label>
						</th>
						<td>
							<select id="mrme-smtp-encryption" name="mrme_smtp_encryption" class="regular-text">
								<option value="tls"  <?php selected( $smtp_enc, 'tls' ); ?>><?php esc_html_e( 'TLS (recommended, port 587)', 'myrock-mail-engine' ); ?></option>
								<option value="ssl"  <?php selected( $smtp_enc, 'ssl' ); ?>><?php esc_html_e( 'SSL (port 465)', 'myrock-mail-engine' ); ?></option>
								<option value="none" <?php selected( $smtp_enc, 'none' ); ?>><?php esc_html_e( 'None (not recommended)', 'myrock-mail-engine' ); ?></option>
							</select>
						</td>
					</tr>
					<tr>
						<th scope="row">
							<label for="mrme-smtp-username"><?php esc_html_e( 'Username', 'myrock-mail-engine' ); ?></label>
						</th>
						<td>
							<input
								type="text"
								id="mrme-smtp-username"
								name="mrme_smtp_username"
								class="regular-text"
								value="<?php echo esc_attr( $smtp_user ); ?>"
								autocomplete="username"
							>
						</td>
					</tr>
					<tr>
						<th scope="row">
							<label for="mrme-smtp-password"><?php esc_html_e( 'Password', 'myrock-mail-engine' ); ?></label>
						</th>
						<td>
							<input
								type="password"
								id="mrme-smtp-password"
								name="mrme_smtp_password"
								class="regular-text"
								value="<?php echo esc_attr( $smtp_pass ); ?>"
								autocomplete="current-password"
							>
							<p class="description"><?php esc_html_e( 'Password is stored encrypted in the database.', 'myrock-mail-engine' ); ?></p>
						</td>
					</tr>
				</tbody>
			</table>

			<p>
				<button type="button" id="mrme-test-smtp-btn" class="button" data-nonce="<?php echo esc_attr( wp_create_nonce( 'mrme_test_smtp' ) ); ?>">
					<?php esc_html_e( 'Test SMTP Connection', 'myrock-mail-engine' ); ?>
				</button>
				<span id="mrme-smtp-test-result" class="mrme-inline-notice" style="display:none;"></span>
			</p>
		</div>

		<!-- ============================================================== -->
		<!-- Section 4: API Key                                              -->
		<!-- ============================================================== -->
		<div class="mrme-form-section mrme-form-section--card">
			<h2 class="mrme-form-section__title"><?php esc_html_e( 'REST API Key', 'myrock-mail-engine' ); ?></h2>
			<p class="description">
				<?php esc_html_e( 'Use this key to authenticate requests to the REST API via the X-MRME-Key HTTP header.', 'myrock-mail-engine' ); ?>
			</p>

			<table class="form-table mrme-settings-table">
				<tbody>
					<tr>
						<th scope="row"><?php esc_html_e( 'API Key', 'myrock-mail-engine' ); ?></th>
						<td>
							<?php if ( $api_key ) : ?>
							<code class="mrme-api-key-display" id="mrme-api-key-value"><?php echo esc_html( $api_key ); ?></code>
							<button type="button" id="mrme-copy-api-key" class="button button-small" data-clipboard-target="#mrme-api-key-value">
								<?php esc_html_e( 'Copy', 'myrock-mail-engine' ); ?>
							</button>
							<?php else : ?>
							<em class="mrme-muted"><?php esc_html_e( 'No API key generated yet.', 'myrock-mail-engine' ); ?></em>
							<?php endif; ?>

							<br><br>
							<button
								type="submit"
								name="save_action"
								value="regenerate_api_key"
								class="button"
								<?php if ( $api_key ) : ?>
								onclick="return confirm('<?php echo esc_js( __( 'Regenerating the API key will invalidate the current key. Continue?', 'myrock-mail-engine' ) ); ?>')"
								<?php endif; ?>
							>
								<?php echo $api_key
									? esc_html__( 'Regenerate API Key', 'myrock-mail-engine' )
									: esc_html__( 'Generate API Key', 'myrock-mail-engine' );
								?>
							</button>
						</td>
					</tr>
				</tbody>
			</table>
		</div>

		<!-- ============================================================== -->
		<!-- Section 5: Danger Zone                                          -->
		<!-- ============================================================== -->
		<div class="mrme-form-section mrme-form-section--card mrme-danger-zone">
			<h2 class="mrme-form-section__title mrme-danger-zone__title"><?php esc_html_e( 'Danger Zone', 'myrock-mail-engine' ); ?></h2>

			<table class="form-table mrme-settings-table">
				<tbody>
					<tr>
						<th scope="row"><?php esc_html_e( 'Delete Data on Uninstall', 'myrock-mail-engine' ); ?></th>
						<td>
							<label>
								<input
									type="checkbox"
									name="mrme_delete_on_uninstall"
									value="1"
									<?php checked( $delete_on_uninstall ); ?>
								>
								<?php esc_html_e( 'Remove all plugin data (contacts, campaigns, lists, logs) when the plugin is deleted.', 'myrock-mail-engine' ); ?>
							</label>
							<p class="description mrme-danger-zone__desc">
								<?php esc_html_e( 'Warning: this action is irreversible. Back up your data before uninstalling.', 'myrock-mail-engine' ); ?>
							</p>
						</td>
					</tr>
				</tbody>
			</table>
		</div>

		<!-- ============================================================== -->
		<!-- Save button                                                      -->
		<!-- ============================================================== -->
		<p class="mrme-settings-submit">
			<button type="submit" name="save_action" value="save" class="button button-primary button-large">
				<?php esc_html_e( 'Save Settings', 'myrock-mail-engine' ); ?>
			</button>
		</p>

	</form>

</div><!-- /.wrap.mrme-wrap -->

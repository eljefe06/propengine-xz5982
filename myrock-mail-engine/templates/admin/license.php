<?php
/**
 * Admin template: License management.
 *
 * Variables available (set by LicensePage::render()):
 *   $status      array  — validated status: [ valid, plan, expires, message ]
 *   $key         string — current stored key (may be empty)
 *   $plan        string — 'free' | 'pro'
 *   $is_pro      bool
 *   $notice_type string — 'activated' | 'deactivated' | 'invalid' | ''
 *
 * @package MyRock\MailEngine
 */

defined( 'ABSPATH' ) || exit;
?>
<div class="wrap mrme-wrap">

	<h1 class="mrme-page-title">
		<?php esc_html_e( 'MyRock Mail Engine — License', 'myrock-mail-engine' ); ?>
	</h1>

	<?php
	// ── Notices ───────────────────────────────────────────────────────────
	if ( 'activated' === $notice_type ) :
	?>
	<div class="notice notice-success is-dismissible">
		<p><?php esc_html_e( 'License activated successfully.', 'myrock-mail-engine' ); ?> 🎉</p>
	</div>
	<?php elseif ( 'deactivated' === $notice_type ) : ?>
	<div class="notice notice-info is-dismissible">
		<p><?php esc_html_e( 'License deactivated.', 'myrock-mail-engine' ); ?></p>
	</div>
	<?php elseif ( 'invalid' === $notice_type ) :
		// phpcs:ignore WordPress.Security.NonceVerification.Recommended
		$err_msg = isset( $_GET['mrme_lic_msg'] ) ? sanitize_text_field( urldecode( $_GET['mrme_lic_msg'] ) ) : '';
	?>
	<div class="notice notice-error is-dismissible">
		<p><?php echo esc_html( $err_msg ?: __( 'Invalid or expired license key.', 'myrock-mail-engine' ) ); ?></p>
	</div>
	<?php endif; ?>

	<!-- ============================================================== -->
	<!-- Plan status card                                                 -->
	<!-- ============================================================== -->
	<div class="mrme-form-section mrme-form-section--card mrme-license-card">

		<div class="mrme-license-plan-header">
			<span class="mrme-license-plan-badge mrme-license-plan-badge--<?php echo esc_attr( $plan ); ?>">
				<?php echo $is_pro
					? esc_html__( 'Pro', 'myrock-mail-engine' )
					: esc_html__( 'Free', 'myrock-mail-engine' );
				?>
			</span>
			<h2 class="mrme-form-section__title" style="display:inline;margin-left:.75rem;">
				<?php
				echo $is_pro
					? esc_html__( 'Pro Plan — Active', 'myrock-mail-engine' )
					: esc_html__( 'Free Plan', 'myrock-mail-engine' );
				?>
			</h2>
		</div>

		<?php if ( $is_pro && ! empty( $status['expires'] ) ) : ?>
		<p class="description">
			<?php
			printf(
				/* translators: %s expiry date */
				esc_html__( 'License valid until: %s', 'myrock-mail-engine' ),
				'<strong>' . esc_html( $status['expires'] ) . '</strong>'
			);
			?>
		</p>
		<?php endif; ?>

		<!-- Feature comparison -->
		<table class="mrme-license-features widefat striped" style="margin-top:1.5rem;max-width:640px;">
			<thead>
				<tr>
					<th><?php esc_html_e( 'Feature', 'myrock-mail-engine' ); ?></th>
					<th style="text-align:center;"><?php esc_html_e( 'Free', 'myrock-mail-engine' ); ?></th>
					<th style="text-align:center;">Pro</th>
				</tr>
			</thead>
			<tbody>
				<tr>
					<td><?php esc_html_e( 'Contacts', 'myrock-mail-engine' ); ?></td>
					<td style="text-align:center;">500</td>
					<td style="text-align:center;">∞</td>
				</tr>
				<tr>
					<td><?php esc_html_e( 'Campaigns', 'myrock-mail-engine' ); ?></td>
					<td style="text-align:center;">∞</td>
					<td style="text-align:center;">∞</td>
				</tr>
				<tr>
					<td>Mailgun</td>
					<td style="text-align:center;">✗</td>
					<td style="text-align:center;">✓</td>
				</tr>
				<tr>
					<td><?php esc_html_e( 'Automations', 'myrock-mail-engine' ); ?></td>
					<td style="text-align:center;">✗</td>
					<td style="text-align:center;">✓</td>
				</tr>
				<tr>
					<td><?php esc_html_e( 'Priority support', 'myrock-mail-engine' ); ?></td>
					<td style="text-align:center;">✗</td>
					<td style="text-align:center;">✓</td>
				</tr>
				<tr>
					<td><?php esc_html_e( 'REST API access', 'myrock-mail-engine' ); ?></td>
					<td style="text-align:center;">✓</td>
					<td style="text-align:center;">✓</td>
				</tr>
			</tbody>
		</table>
	</div>

	<!-- ============================================================== -->
	<!-- License key form                                                 -->
	<!-- ============================================================== -->
	<div class="mrme-form-section mrme-form-section--card">
		<h2 class="mrme-form-section__title">
			<?php $is_pro
				? esc_html_e( 'Manage License Key', 'myrock-mail-engine' )
				: esc_html_e( 'Activate License Key', 'myrock-mail-engine' );
			?>
		</h2>

		<?php if ( ! $is_pro ) : ?>
		<p class="description">
			<?php esc_html_e( 'Get your license at', 'myrock-mail-engine' ); ?>
			<a href="https://myrock.com.mx/plugin" target="_blank" rel="noopener"><strong>myrock.com.mx/plugin</strong></a>.
		</p>
		<?php endif; ?>

		<form method="POST" action="<?php echo esc_url( admin_url( 'admin-post.php' ) ); ?>">
			<?php wp_nonce_field( 'mrme_save_license' ); ?>
			<input type="hidden" name="action" value="mrme_save_license">

			<table class="form-table mrme-settings-table">
				<tbody>
					<tr>
						<th scope="row">
							<label for="mrme-license-key"><?php esc_html_e( 'License Key', 'myrock-mail-engine' ); ?></label>
						</th>
						<td>
							<input
								type="text"
								id="mrme-license-key"
								name="mrme_license_key"
								class="regular-text"
								value="<?php echo esc_attr( $is_pro ? substr( $key, 0, 8 ) . str_repeat( '•', max( 0, strlen( $key ) - 8 ) ) : $key ); ?>"
								placeholder="MRME-XXXX-XXXX-XXXX-XXXX"
								<?php echo $is_pro ? 'readonly' : ''; ?>
								autocomplete="off"
								spellcheck="false"
							>
							<?php if ( $is_pro ) : ?>
							<p class="description"><?php esc_html_e( 'Key is active. Deactivate below to change it.', 'myrock-mail-engine' ); ?></p>
							<?php endif; ?>
						</td>
					</tr>
				</tbody>
			</table>

			<p class="mrme-settings-submit">
				<?php if ( ! $is_pro ) : ?>
				<button type="submit" name="license_action" value="activate" class="button button-primary button-large">
					<?php esc_html_e( 'Activate License', 'myrock-mail-engine' ); ?>
				</button>
				<?php else : ?>
				<button
					type="submit"
					name="license_action"
					value="deactivate"
					class="button button-secondary"
					onclick="return confirm('<?php echo esc_js( __( 'Deactivating will revert to the Free plan. Continue?', 'myrock-mail-engine' ) ); ?>')"
				>
					<?php esc_html_e( 'Deactivate', 'myrock-mail-engine' ); ?>
				</button>
				<?php endif; ?>
			</p>
		</form>
	</div>

	<?php if ( ! $is_pro ) : ?>
	<!-- ============================================================== -->
	<!-- Upgrade CTA                                                      -->
	<!-- ============================================================== -->
	<div class="mrme-form-section mrme-form-section--card" style="background:linear-gradient(135deg,#1B2980 0%,#2563EB 100%);color:#fff;border:none;">
		<h2 style="color:#fff;margin-top:0;"><?php esc_html_e( 'Upgrade to Pro', 'myrock-mail-engine' ); ?> 🚀</h2>
		<p style="color:rgba(255,255,255,.85);font-size:1rem;max-width:540px;">
			<?php esc_html_e( 'Unlock unlimited contacts, Mailgun integration, automations and priority support.', 'myrock-mail-engine' ); ?>
		</p>
		<a
			href="https://myrock.com.mx/plugin"
			target="_blank"
			rel="noopener"
			class="button button-large"
			style="background:#fff;color:#1B2980;border-color:#fff;font-weight:600;margin-top:.5rem;"
		>
			<?php esc_html_e( 'Get Pro License', 'myrock-mail-engine' ); ?> →
		</a>
	</div>
	<?php endif; ?>

</div><!-- /.wrap.mrme-wrap -->

<style>
.mrme-license-plan-badge {
	display: inline-block;
	padding: 3px 14px;
	border-radius: 100px;
	font-size: .75rem;
	font-weight: 700;
	letter-spacing: .08em;
	text-transform: uppercase;
	vertical-align: middle;
}
.mrme-license-plan-badge--free {
	background: #f1f5f9;
	color: #64748b;
	border: 1px solid #cbd5e1;
}
.mrme-license-plan-badge--pro {
	background: #1B2980;
	color: #fff;
}
.mrme-license-features td,
.mrme-license-features th {
	padding: 10px 14px;
}
</style>

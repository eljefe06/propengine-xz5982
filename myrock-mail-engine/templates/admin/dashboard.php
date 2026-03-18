<?php
/**
 * Admin template: Dashboard overview.
 *
 * Available variables (set by DashboardPage before including this template):
 *   @var int   $total_contacts   Total number of contacts in the database.
 *   @var int   $total_campaigns  Total number of campaigns.
 *   @var int   $total_sent       Total emails sent across all campaigns.
 *   @var int   $total_opens      Total email opens tracked.
 *   @var array $recent_campaigns Last 5 campaigns (each has: id, title, status, sent_at, stats).
 *
 * @package MyRock\MailEngine
 */

defined( 'ABSPATH' ) || exit;
?>
<div class="wrap mrme-wrap">

	<h1 class="mrme-page-title">
		<span class="mrme-logo-mark">&#9679;</span>
		<?php esc_html_e( 'MyRock Mail Engine — Dashboard', 'myrock-mail-engine' ); ?>
	</h1>

	<?php
	$_lic = \MyRock\MailEngine\License\LicenseManager::get_status();
	if ( ! \MyRock\MailEngine\License\LicenseManager::is_pro() ) :
	?>
	<div class="notice notice-warning" style="padding:10px 14px;display:flex;align-items:center;gap:14px;border-left-color:#1B2980;">
		<strong style="color:#1B2980;">FREE</strong>
		<span>
			<?php esc_html_e( 'You are on the Free plan. Mailgun and Automations are locked.', 'myrock-mail-engine' ); ?>
			&nbsp;<a href="<?php echo esc_url( admin_url( 'admin.php?page=mrme-license' ) ); ?>" style="font-weight:600;">
				<?php esc_html_e( 'Upgrade to Pro →', 'myrock-mail-engine' ); ?>
			</a>
		</span>
	</div>
	<?php endif; ?>

	<!-- ================================================================ -->
	<!-- Stat cards                                                        -->
	<!-- ================================================================ -->
	<div class="mrme-stat-cards">

		<div class="mrme-stat-card">
			<div class="mrme-stat-card__icon dashicons dashicons-groups"></div>
			<div class="mrme-stat-card__value"><?php echo esc_html( number_format_i18n( $total_contacts ) ); ?></div>
			<div class="mrme-stat-card__label"><?php esc_html_e( 'Total Contacts', 'myrock-mail-engine' ); ?></div>
			<a class="mrme-stat-card__link" href="<?php echo esc_url( admin_url( 'admin.php?page=mrme-contacts' ) ); ?>">
				<?php esc_html_e( 'View all →', 'myrock-mail-engine' ); ?>
			</a>
		</div>

		<div class="mrme-stat-card">
			<div class="mrme-stat-card__icon dashicons dashicons-email-alt"></div>
			<div class="mrme-stat-card__value"><?php echo esc_html( number_format_i18n( $total_campaigns ) ); ?></div>
			<div class="mrme-stat-card__label"><?php esc_html_e( 'Campaigns Sent', 'myrock-mail-engine' ); ?></div>
			<a class="mrme-stat-card__link" href="<?php echo esc_url( admin_url( 'admin.php?page=mrme-campaigns' ) ); ?>">
				<?php esc_html_e( 'View all →', 'myrock-mail-engine' ); ?>
			</a>
		</div>

		<div class="mrme-stat-card">
			<div class="mrme-stat-card__icon dashicons dashicons-chart-bar"></div>
			<div class="mrme-stat-card__value"><?php echo esc_html( number_format_i18n( $total_sent ) ); ?></div>
			<div class="mrme-stat-card__label"><?php esc_html_e( 'Emails Sent', 'myrock-mail-engine' ); ?></div>
		</div>

		<div class="mrme-stat-card">
			<div class="mrme-stat-card__icon dashicons dashicons-visibility"></div>
			<div class="mrme-stat-card__value"><?php echo esc_html( number_format_i18n( $total_opens ) ); ?></div>
			<div class="mrme-stat-card__label"><?php esc_html_e( 'Opens', 'myrock-mail-engine' ); ?></div>
		</div>

	</div><!-- /.mrme-stat-cards -->

	<!-- ================================================================ -->
	<!-- Recent campaigns                                                  -->
	<!-- ================================================================ -->
	<div class="mrme-section">
		<div class="mrme-section__header">
			<h2><?php esc_html_e( 'Recent Campaigns', 'myrock-mail-engine' ); ?></h2>
			<a href="<?php echo esc_url( admin_url( 'admin.php?page=mrme-campaigns&action=new' ) ); ?>" class="button button-primary">
				<?php esc_html_e( '+ Create Campaign', 'myrock-mail-engine' ); ?>
			</a>
		</div>

		<?php if ( ! empty( $recent_campaigns ) ) : ?>
		<table class="mrme-table widefat">
			<thead>
				<tr>
					<th><?php esc_html_e( 'Title', 'myrock-mail-engine' ); ?></th>
					<th><?php esc_html_e( 'Status', 'myrock-mail-engine' ); ?></th>
					<th><?php esc_html_e( 'Sent', 'myrock-mail-engine' ); ?></th>
					<th><?php esc_html_e( 'Opens', 'myrock-mail-engine' ); ?></th>
					<th><?php esc_html_e( 'Clicks', 'myrock-mail-engine' ); ?></th>
					<th><?php esc_html_e( 'Date', 'myrock-mail-engine' ); ?></th>
					<th><?php esc_html_e( 'Actions', 'myrock-mail-engine' ); ?></th>
				</tr>
			</thead>
			<tbody>
				<?php foreach ( $recent_campaigns as $campaign ) :
					$c_id     = is_array( $campaign ) ? $campaign['id']     : $campaign->id;
					$c_title  = is_array( $campaign ) ? $campaign['title']  : $campaign->title;
					$c_status = is_array( $campaign ) ? $campaign['status'] : $campaign->status;
					$c_sent   = is_array( $campaign ) ? ( $campaign['total_sent']   ?? 0 ) : ( $campaign->total_sent   ?? 0 );
					$c_opens  = is_array( $campaign ) ? ( $campaign['total_opens']  ?? 0 ) : ( $campaign->total_opens  ?? 0 );
					$c_clicks = is_array( $campaign ) ? ( $campaign['total_clicks'] ?? 0 ) : ( $campaign->total_clicks ?? 0 );
					$c_date   = is_array( $campaign ) ? ( $campaign['sent_at'] ?? ( $campaign['scheduled_at'] ?? '' ) ) : ( $campaign->sent_at ?? ( $campaign->scheduled_at ?? '' ) );
					?>
				<tr>
					<td>
						<strong>
							<a href="<?php echo esc_url( admin_url( 'admin.php?page=mrme-campaigns&action=edit&id=' . (int) $c_id ) ); ?>">
								<?php echo esc_html( $c_title ); ?>
							</a>
						</strong>
					</td>
					<td>
						<span class="mrme-badge mrme-badge--<?php echo esc_attr( $c_status ); ?>">
							<?php echo esc_html( ucfirst( $c_status ) ); ?>
						</span>
					</td>
					<td><?php echo esc_html( number_format_i18n( (int) $c_sent ) ); ?></td>
					<td><?php echo esc_html( number_format_i18n( (int) $c_opens ) ); ?></td>
					<td><?php echo esc_html( number_format_i18n( (int) $c_clicks ) ); ?></td>
					<td><?php echo esc_html( $c_date ? wp_date( get_option( 'date_format' ), strtotime( $c_date ) ) : '—' ); ?></td>
					<td>
						<a href="<?php echo esc_url( admin_url( 'admin.php?page=mrme-campaigns&action=edit&id=' . (int) $c_id ) ); ?>">
							<?php esc_html_e( 'Edit', 'myrock-mail-engine' ); ?>
						</a>
					</td>
				</tr>
				<?php endforeach; ?>
			</tbody>
		</table>
		<?php else : ?>
		<div class="mrme-empty-state">
			<span class="dashicons dashicons-email-alt mrme-empty-state__icon"></span>
			<p><?php esc_html_e( 'No campaigns yet. Create your first campaign to get started.', 'myrock-mail-engine' ); ?></p>
			<a href="<?php echo esc_url( admin_url( 'admin.php?page=mrme-campaigns&action=new' ) ); ?>" class="button button-primary">
				<?php esc_html_e( '+ Create Campaign', 'myrock-mail-engine' ); ?>
			</a>
		</div>
		<?php endif; ?>
	</div><!-- /.mrme-section -->

	<!-- ================================================================ -->
	<!-- Quick links                                                       -->
	<!-- ================================================================ -->
	<div class="mrme-section mrme-quick-links">
		<h2><?php esc_html_e( 'Quick Actions', 'myrock-mail-engine' ); ?></h2>
		<div class="mrme-quick-links__grid">

			<a href="<?php echo esc_url( admin_url( 'admin.php?page=mrme-contacts&action=new' ) ); ?>" class="mrme-quick-link-card">
				<span class="dashicons dashicons-plus-alt mrme-quick-link-card__icon"></span>
				<span><?php esc_html_e( 'Add Contact', 'myrock-mail-engine' ); ?></span>
			</a>

			<a href="<?php echo esc_url( admin_url( 'admin.php?page=mrme-campaigns&action=new' ) ); ?>" class="mrme-quick-link-card">
				<span class="dashicons dashicons-email mrme-quick-link-card__icon"></span>
				<span><?php esc_html_e( 'Create Campaign', 'myrock-mail-engine' ); ?></span>
			</a>

			<a href="<?php echo esc_url( admin_url( 'admin.php?page=mrme-contacts&action=import' ) ); ?>" class="mrme-quick-link-card">
				<span class="dashicons dashicons-upload mrme-quick-link-card__icon"></span>
				<span><?php esc_html_e( 'Import Contacts', 'myrock-mail-engine' ); ?></span>
			</a>

			<a href="<?php echo esc_url( admin_url( 'admin.php?page=mrme-settings' ) ); ?>" class="mrme-quick-link-card">
				<span class="dashicons dashicons-admin-settings mrme-quick-link-card__icon"></span>
				<span><?php esc_html_e( 'Settings', 'myrock-mail-engine' ); ?></span>
			</a>

		</div>
	</div><!-- /.mrme-quick-links -->

</div><!-- /.wrap.mrme-wrap -->

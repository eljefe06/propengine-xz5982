<?php
/**
 * Admin template: Send Logs list.
 *
 * Available variables (set by LogsPage before including this template):
 *   @var array  $logs             Paged array of send-log rows.
 *   @var int    $total            Total log count for current filter.
 *   @var int    $paged            Current page number.
 *   @var int    $per_page         Rows per page (default: 50).
 *   @var array  $campaigns        All campaigns for the filter dropdown.
 *   @var int    $campaign_filter  Active campaign ID filter (0 = all).
 *   @var string $status_filter    Active status tab ('', 'sent', 'failed', …).
 *
 * @package MyRock\MailEngine
 */

defined( 'ABSPATH' ) || exit;

$total_pages    = $per_page > 0 ? (int) ceil( $total / $per_page ) : 1;
$base_url       = admin_url( 'admin.php?page=mrme-logs' );

$status_tabs = [
	''        => __( 'All', 'myrock-mail-engine' ),
	'sent'    => __( 'Sent', 'myrock-mail-engine' ),
	'failed'  => __( 'Failed', 'myrock-mail-engine' ),
	'bounced' => __( 'Bounced', 'myrock-mail-engine' ),
	'opened'  => __( 'Opened', 'myrock-mail-engine' ),
	'clicked' => __( 'Clicked', 'myrock-mail-engine' ),
];
?>
<div class="wrap mrme-wrap">

	<!-- Page header -->
	<div class="mrme-page-header">
		<h1 class="mrme-page-title"><?php esc_html_e( 'Send Logs', 'myrock-mail-engine' ); ?></h1>
	</div>

	<!-- Filters -->
	<div class="mrme-filter-bar">

		<!-- Status tabs -->
		<ul class="mrme-status-tabs">
			<?php foreach ( $status_tabs as $tab_key => $tab_label ) :
				$tab_url    = $tab_key
					? add_query_arg( 'status', $tab_key, $base_url )
					: $base_url;
				if ( $campaign_filter ) {
					$tab_url = add_query_arg( 'campaign_id', $campaign_filter, $tab_url );
				}
				$tab_active = ( $status_filter === $tab_key ) ? 'mrme-status-tabs__item--active' : '';
			?>
			<li class="mrme-status-tabs__item <?php echo esc_attr( $tab_active ); ?>">
				<a href="<?php echo esc_url( $tab_url ); ?>"><?php echo esc_html( $tab_label ); ?></a>
			</li>
			<?php endforeach; ?>
		</ul>

		<!-- Campaign dropdown filter -->
		<form method="GET" class="mrme-logs-filter" action="<?php echo esc_url( $base_url ); ?>">
			<input type="hidden" name="page" value="mrme-logs">
			<?php if ( $status_filter ) : ?>
			<input type="hidden" name="status" value="<?php echo esc_attr( $status_filter ); ?>">
			<?php endif; ?>
			<label for="mrme-log-campaign-filter" class="screen-reader-text">
				<?php esc_html_e( 'Filter by Campaign', 'myrock-mail-engine' ); ?>
			</label>
			<select id="mrme-log-campaign-filter" name="campaign_id" class="mrme-logs-filter__select">
				<option value=""><?php esc_html_e( '— All Campaigns —', 'myrock-mail-engine' ); ?></option>
				<?php foreach ( $campaigns as $campaign ) :
					$c_id    = is_array( $campaign ) ? $campaign['id']    : $campaign->id;
					$c_title = is_array( $campaign ) ? $campaign['title'] : $campaign->title;
				?>
				<option value="<?php echo esc_attr( (int) $c_id ); ?>" <?php selected( $campaign_filter, (int) $c_id ); ?>>
					<?php echo esc_html( $c_title ); ?>
				</option>
				<?php endforeach; ?>
			</select>
			<button type="submit" class="button"><?php esc_html_e( 'Filter', 'myrock-mail-engine' ); ?></button>
			<?php if ( $campaign_filter ) : ?>
			<a href="<?php echo esc_url( $base_url . ( $status_filter ? '&status=' . esc_attr( $status_filter ) : '' ) ); ?>" class="button">
				<?php esc_html_e( 'Clear', 'myrock-mail-engine' ); ?>
			</a>
			<?php endif; ?>
		</form>

	</div><!-- /.mrme-filter-bar -->

	<!-- Results summary -->
	<p class="mrme-results-summary">
		<?php
		printf(
			esc_html__( 'Showing %1$s of %2$s log entries', 'myrock-mail-engine' ),
			'<strong>' . esc_html( number_format_i18n( count( $logs ) ) ) . '</strong>',
			'<strong>' . esc_html( number_format_i18n( $total ) ) . '</strong>'
		);
		?>
	</p>

	<?php if ( ! empty( $logs ) ) : ?>

	<table class="mrme-table widefat">
		<thead>
			<tr>
				<th><?php esc_html_e( 'ID', 'myrock-mail-engine' ); ?></th>
				<th><?php esc_html_e( 'Campaign', 'myrock-mail-engine' ); ?></th>
				<th><?php esc_html_e( 'Email', 'myrock-mail-engine' ); ?></th>
				<th><?php esc_html_e( 'Status', 'myrock-mail-engine' ); ?></th>
				<th><?php esc_html_e( 'Sent At', 'myrock-mail-engine' ); ?></th>
				<th><?php esc_html_e( 'Opened At', 'myrock-mail-engine' ); ?></th>
				<th><?php esc_html_e( 'Error', 'myrock-mail-engine' ); ?></th>
			</tr>
		</thead>
		<tbody>
			<?php foreach ( $logs as $log ) :
				$l_id          = is_array( $log ) ? $log['id']              : $log->id;
				$l_campaign_id = is_array( $log ) ? ( $log['campaign_id']   ?? 0 )  : ( $log->campaign_id   ?? 0 );
				$l_campaign    = is_array( $log ) ? ( $log['campaign_title'] ?? '' ) : ( $log->campaign_title ?? '' );
				$l_email       = is_array( $log ) ? $log['email']           : $log->email;
				$l_status      = is_array( $log ) ? $log['status']          : $log->status;
				$l_sent_at     = is_array( $log ) ? ( $log['sent_at']    ?? '' ) : ( $log->sent_at    ?? '' );
				$l_opened_at   = is_array( $log ) ? ( $log['opened_at']  ?? '' ) : ( $log->opened_at  ?? '' );
				$l_error       = is_array( $log ) ? ( $log['error']      ?? '' ) : ( $log->error      ?? '' );

				$date_format   = get_option( 'date_format' ) . ' H:i:s';
			?>
			<tr>
				<td class="mrme-muted"><?php echo esc_html( (int) $l_id ); ?></td>
				<td>
					<?php if ( $l_campaign_id ) : ?>
					<a href="<?php echo esc_url( admin_url( 'admin.php?page=mrme-campaigns&action=edit&id=' . (int) $l_campaign_id ) ); ?>">
						<?php echo esc_html( $l_campaign ?: '#' . (int) $l_campaign_id ); ?>
					</a>
					<?php else : ?>
					<span class="mrme-muted">—</span>
					<?php endif; ?>
				</td>
				<td><?php echo esc_html( $l_email ); ?></td>
				<td>
					<?php
					// Map log status to badge modifier.
					$badge_map = [
						'sent'    => 'sent',
						'failed'  => 'bounced',
						'bounced' => 'bounced',
						'opened'  => 'subscribed',
						'clicked' => 'info',
						'pending' => 'pending',
					];
					$badge_mod = $badge_map[ $l_status ] ?? 'draft';
					?>
					<span class="mrme-badge mrme-badge--<?php echo esc_attr( $badge_mod ); ?>">
						<?php echo esc_html( ucfirst( $l_status ) ); ?>
					</span>
				</td>
				<td>
					<?php echo esc_html( $l_sent_at
						? wp_date( $date_format, strtotime( $l_sent_at ) )
						: '—'
					); ?>
				</td>
				<td>
					<?php echo esc_html( $l_opened_at
						? wp_date( $date_format, strtotime( $l_opened_at ) )
						: '—'
					); ?>
				</td>
				<td>
					<?php if ( $l_error ) : ?>
					<span class="mrme-log-error-text" title="<?php echo esc_attr( $l_error ); ?>">
						<?php echo esc_html( mb_strimwidth( $l_error, 0, 80, '…' ) ); ?>
					</span>
					<?php else : ?>
					<span class="mrme-muted">—</span>
					<?php endif; ?>
				</td>
			</tr>
			<?php endforeach; ?>
		</tbody>
	</table>

	<?php else : ?>
	<div class="mrme-empty-state">
		<span class="dashicons dashicons-list-view mrme-empty-state__icon"></span>
		<p>
			<?php esc_html_e( 'No log entries found for the current filter.', 'myrock-mail-engine' ); ?>
		</p>
	</div>
	<?php endif; ?>

	<!-- Pagination -->
	<?php if ( $total_pages > 1 ) : ?>
	<div class="mrme-pagination">
		<?php
		$paginate_base = add_query_arg( 'paged', '%#%', $base_url );
		if ( $status_filter ) {
			$paginate_base = add_query_arg( 'status', $status_filter, $paginate_base );
		}
		if ( $campaign_filter ) {
			$paginate_base = add_query_arg( 'campaign_id', $campaign_filter, $paginate_base );
		}
		echo wp_kses_post( paginate_links( [
			'base'      => $paginate_base,
			'format'    => '',
			'current'   => $paged,
			'total'     => $total_pages,
			'prev_text' => '&laquo; ' . esc_html__( 'Previous', 'myrock-mail-engine' ),
			'next_text' => esc_html__( 'Next', 'myrock-mail-engine' ) . ' &raquo;',
		] ) );
		?>
	</div>
	<?php endif; ?>

</div><!-- /.wrap.mrme-wrap -->

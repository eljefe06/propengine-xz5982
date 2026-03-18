<?php

namespace MyRock\LicenseServer\Models;

defined( 'ABSPATH' ) || exit;

class License {

	public int     $id               = 0;
	public string  $license_key      = '';
	public string  $email            = '';
	public string  $plan             = 'monthly';
	public string  $status           = 'pending';
	public ?string $mp_subscription_id = null;
	public ?string $mp_payer_email   = null;
	public ?string $site_url         = null;
	public string  $created_at       = '';
	public ?string $expires_at       = null;
	public ?string $last_renewed_at  = null;
	public ?string $notes            = null;

	// -------------------------------------------------------------------------
	// Finders
	// -------------------------------------------------------------------------

	public static function find( int $id ): ?self {
		global $wpdb;
		$row = $wpdb->get_row( $wpdb->prepare(
			"SELECT * FROM {$wpdb->prefix}mrls_licenses WHERE id = %d",
			$id
		) );
		return $row ? self::from_row( $row ) : null;
	}

	public static function find_by_key( string $key ): ?self {
		global $wpdb;
		$row = $wpdb->get_row( $wpdb->prepare(
			"SELECT * FROM {$wpdb->prefix}mrls_licenses WHERE license_key = %s",
			$key
		) );
		return $row ? self::from_row( $row ) : null;
	}

	public static function find_by_subscription( string $mp_id ): ?self {
		global $wpdb;
		$row = $wpdb->get_row( $wpdb->prepare(
			"SELECT * FROM {$wpdb->prefix}mrls_licenses WHERE mp_subscription_id = %s",
			$mp_id
		) );
		return $row ? self::from_row( $row ) : null;
	}

	/**
	 * @return self[]
	 */
	public static function find_all( int $per_page = 50, int $page = 1 ): array {
		global $wpdb;
		$offset = ( $page - 1 ) * $per_page;
		$rows   = $wpdb->get_results( $wpdb->prepare(
			"SELECT * FROM {$wpdb->prefix}mrls_licenses ORDER BY created_at DESC LIMIT %d OFFSET %d",
			$per_page,
			$offset
		) );
		return array_map( [ self::class, 'from_row' ], $rows );
	}

	public static function count(): int {
		global $wpdb;
		return (int) $wpdb->get_var( "SELECT COUNT(*) FROM {$wpdb->prefix}mrls_licenses" );
	}

	// -------------------------------------------------------------------------
	// Persistence
	// -------------------------------------------------------------------------

	public function save(): bool {
		global $wpdb;

		$data = [
			'license_key'        => $this->license_key,
			'email'              => $this->email,
			'plan'               => $this->plan,
			'status'             => $this->status,
			'mp_subscription_id' => $this->mp_subscription_id,
			'mp_payer_email'     => $this->mp_payer_email,
			'site_url'           => $this->site_url,
			'created_at'         => $this->created_at,
			'expires_at'         => $this->expires_at,
			'last_renewed_at'    => $this->last_renewed_at,
			'notes'              => $this->notes,
		];

		if ( $this->id ) {
			$result = $wpdb->update( "{$wpdb->prefix}mrls_licenses", $data, [ 'id' => $this->id ] );
		} else {
			$result = $wpdb->insert( "{$wpdb->prefix}mrls_licenses", $data );
			if ( $result ) {
				$this->id = $wpdb->insert_id;
			}
		}

		return false !== $result;
	}

	// -------------------------------------------------------------------------
	// Business logic
	// -------------------------------------------------------------------------

	public function is_active(): bool {
		if ( 'active' !== $this->status ) {
			return false;
		}
		if ( $this->expires_at && strtotime( $this->expires_at ) < time() ) {
			return false;
		}
		return true;
	}

	// -------------------------------------------------------------------------
	// Factory
	// -------------------------------------------------------------------------

	private static function from_row( object $row ): self {
		$obj = new self();
		foreach ( get_object_vars( $row ) as $k => $v ) {
			if ( property_exists( $obj, $k ) ) {
				$obj->$k = $v;
			}
		}
		$obj->id = (int) $row->id;
		return $obj;
	}
}

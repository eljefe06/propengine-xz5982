<?php
namespace MyRock\MailEngine\Services;

defined( 'ABSPATH' ) || exit;

class CsvImporter {

	/**
	 * Maximum number of data rows to read from a CSV file (safety limit).
	 */
	const MAX_ROWS = 10000;

	/**
	 * Parse a CSV file into headers and data rows.
	 *
	 * Auto-detects whether the delimiter is a comma or a semicolon by sampling
	 * the first non-empty line. Headers are lowercased and trimmed. Returns at
	 * most MAX_ROWS data rows.
	 *
	 * @param string $file_path  Absolute path to the CSV file.
	 * @return array {
	 *     @type string[] $headers  Lowercased, trimmed header names.
	 *     @type array[]  $rows     Indexed array of value arrays (one per data row).
	 * }
	 */
	public static function parse( string $file_path ): array {
		$result = [
			'headers' => [],
			'rows'    => [],
		];

		if ( ! file_exists( $file_path ) || ! is_readable( $file_path ) ) {
			return $result;
		}

		$handle = fopen( $file_path, 'r' );
		if ( false === $handle ) {
			return $result;
		}

		// Detect delimiter by reading the first non-empty line and counting
		// the occurrence of ',' vs ';'.
		$first_line = '';
		while ( ! feof( $handle ) ) {
			$line = fgets( $handle );
			if ( false === $line ) {
				break;
			}
			$trimmed = trim( $line );
			if ( $trimmed !== '' ) {
				$first_line = $trimmed;
				break;
			}
		}

		// Default to comma; switch to semicolon only when semicolons clearly
		// outnumber commas on the first line.
		$comma_count     = substr_count( $first_line, ',' );
		$semicolon_count = substr_count( $first_line, ';' );
		$separator       = ( $semicolon_count > $comma_count ) ? ';' : ',';

		// Rewind so we can read from the beginning again with fgetcsv.
		rewind( $handle );

		// Skip BOM if present (UTF-8 BOM: 0xEF 0xBB 0xBF).
		$bom = fread( $handle, 3 );
		if ( $bom !== "\xEF\xBB\xBF" ) {
			rewind( $handle );
		}

		// Read header row — skip blank lines.
		$raw_headers = null;
		while ( ! feof( $handle ) ) {
			$row = fgetcsv( $handle, 0, $separator );
			if ( false === $row || null === $row ) {
				continue;
			}
			// Skip entirely empty rows.
			if ( count( $row ) === 1 && $row[0] === null ) {
				continue;
			}
			$raw_headers = $row;
			break;
		}

		if ( null === $raw_headers ) {
			fclose( $handle );
			return $result;
		}

		// Lowercase and trim each header value.
		$result['headers'] = array_map( static fn( $h ) => strtolower( trim( (string) $h ) ), $raw_headers );

		// Read data rows up to MAX_ROWS.
		$row_count = 0;
		while ( ! feof( $handle ) && $row_count < self::MAX_ROWS ) {
			$row = fgetcsv( $handle, 0, $separator );

			if ( false === $row || null === $row ) {
				continue;
			}

			// Skip entirely empty rows.
			if ( count( $row ) === 1 && $row[0] === null ) {
				continue;
			}

			$result['rows'][] = $row;
			$row_count++;
		}

		fclose( $handle );

		return $result;
	}

	/**
	 * Normalize an array of raw CSV header names to canonical field names.
	 *
	 * Supports common international variations:
	 *   email        ← email, e-mail, correo
	 *   first_name   ← first_name, firstname, nombre, "first name"
	 *   last_name    ← last_name, lastname, apellido, "last name"
	 *   phone        ← phone, telefono, teléfono, tel
	 *   company      ← company, empresa, organización, organizacion
	 *
	 * Unrecognised headers are returned as-is.
	 *
	 * @param string[] $headers  Lowercased, trimmed header strings.
	 * @return string[]          Array of canonical field names with the same count/order.
	 */
	public static function normalize_headers( array $headers ): array {
		// Map of known aliases → canonical name.
		$alias_map = [
			// email
			'email'         => 'email',
			'e-mail'        => 'email',
			'correo'        => 'email',
			'mail'          => 'email',

			// first_name
			'first_name'    => 'first_name',
			'firstname'     => 'first_name',
			'first name'    => 'first_name',
			'nombre'        => 'first_name',
			'given_name'    => 'first_name',
			'given name'    => 'first_name',
			'prenom'        => 'first_name',
			'prénom'        => 'first_name',
			'vorname'       => 'first_name',

			// last_name
			'last_name'     => 'last_name',
			'lastname'      => 'last_name',
			'last name'     => 'last_name',
			'apellido'      => 'last_name',
			'surname'       => 'last_name',
			'family_name'   => 'last_name',
			'family name'   => 'last_name',
			'nom'           => 'last_name',
			'nachname'      => 'last_name',

			// phone
			'phone'         => 'phone',
			'phone_number'  => 'phone',
			'phone number'  => 'phone',
			'telefono'      => 'phone',
			'teléfono'      => 'phone',
			'tel'           => 'phone',
			'telephone'     => 'phone',
			'mobile'        => 'phone',
			'celular'       => 'phone',

			// company
			'company'       => 'company',
			'empresa'       => 'company',
			'organización'  => 'company',
			'organizacion'  => 'company',
			'organisation'  => 'company',
			'organization'  => 'company',
			'company_name'  => 'company',
			'company name'  => 'company',
			'firma'         => 'company',
		];

		$normalized = [];
		foreach ( $headers as $header ) {
			$key              = strtolower( trim( $header ) );
			$normalized[]     = $alias_map[ $key ] ?? $header;
		}

		return $normalized;
	}

	/**
	 * Map a single CSV row's values to an associative array keyed by canonical field names.
	 *
	 * Values are trimmed but not otherwise sanitized (sanitization is the
	 * caller's responsibility).
	 *
	 * If a row has fewer values than headers, missing positions are treated as
	 * empty strings. Extra values beyond the header count are discarded.
	 *
	 * @param string[] $headers  Canonical (already normalized) header names.
	 * @param string[] $row      Raw values from fgetcsv for one data row.
	 * @return array             Associative array: canonical_field_name => value.
	 */
	public static function map_row( array $headers, array $row ): array {
		$mapped = [];

		foreach ( $headers as $index => $field ) {
			$raw_value     = $row[ $index ] ?? '';
			$mapped[ $field ] = trim( (string) $raw_value );
		}

		return $mapped;
	}
}

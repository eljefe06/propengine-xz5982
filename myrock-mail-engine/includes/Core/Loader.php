<?php
/**
 * Registers hooks and filters in an organized manner.
 *
 * @package MyRock\MailEngine\Core
 */

namespace MyRock\MailEngine\Core;

defined( 'ABSPATH' ) || exit;

class Loader {

	/** @var array<array{hook:string, component:object, callback:string, priority:int, args:int}> */
	private array $actions = [];

	/** @var array<array{hook:string, component:object, callback:string, priority:int, args:int}> */
	private array $filters = [];

	public function add_action( string $hook, object $component, string $callback, int $priority = 10, int $args = 1 ): void {
		$this->actions[] = compact( 'hook', 'component', 'callback', 'priority', 'args' );
	}

	public function add_filter( string $hook, object $component, string $callback, int $priority = 10, int $args = 1 ): void {
		$this->filters[] = compact( 'hook', 'component', 'callback', 'priority', 'args' );
	}

	public function run(): void {
		foreach ( $this->actions as $action ) {
			add_action(
				$action['hook'],
				[ $action['component'], $action['callback'] ],
				$action['priority'],
				$action['args']
			);
		}
		foreach ( $this->filters as $filter ) {
			add_filter(
				$filter['hook'],
				[ $filter['component'], $filter['callback'] ],
				$filter['priority'],
				$filter['args']
			);
		}
	}
}

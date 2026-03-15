<?php
/**
 * Site header template part.
 *
 * @package MyRock_Theme
 */
defined( 'ABSPATH' ) || exit;
?>
<header id="mr-site-header" class="mr-site-header" role="banner">
  <div class="mr-container mr-header-inner">

    <a href="<?php echo esc_url( home_url( '/' ) ); ?>" class="mr-site-logo" aria-label="MyRock — Inicio">
      <div class="mr-logo-mark" aria-hidden="true">MR</div>
      <span class="mr-logo-name">MyRock</span>
    </a>

    <nav class="mr-header-nav" aria-label="Navegación principal">
      <a href="<?php echo esc_url( home_url( '/' ) ); ?>">Inicio</a>
      <a href="<?php echo esc_url( home_url( '/blog/' ) ); ?>" <?php echo is_home() || is_archive() || is_single() ? 'aria-current="page"' : ''; ?>>Blog</a>
      <a href="<?php echo esc_url( home_url( '/#services' ) ); ?>">Servicios</a>
      <a href="<?php echo esc_url( home_url( '/#contact' ) ); ?>">Contacto</a>
    </nav>

    <a href="<?php echo esc_url( home_url( '/#contact' ) ); ?>" class="mr-header-cta">Hablemos</a>

    <button class="mr-menu-btn" aria-label="Abrir menú" id="mrMenuBtn" aria-expanded="false">
      <svg width="22" height="16" viewBox="0 0 22 16" fill="none" aria-hidden="true">
        <line x1="0" y1="1"  x2="22" y2="1"  stroke="currentColor" stroke-width="1.5"/>
        <line x1="0" y1="8"  x2="22" y2="8"  stroke="currentColor" stroke-width="1.5"/>
        <line x1="0" y1="15" x2="22" y2="15" stroke="currentColor" stroke-width="1.5"/>
      </svg>
    </button>

  </div>
</header>
<script>
(function() {
  const header = document.getElementById('mr-site-header');
  window.addEventListener('scroll', () => header.classList.toggle('scrolled', window.scrollY > 40), { passive: true });

  const btn = document.getElementById('mrMenuBtn');
  const nav = document.querySelector('.mr-header-nav');
  btn?.addEventListener('click', () => {
    const open = btn.getAttribute('aria-expanded') === 'true';
    btn.setAttribute('aria-expanded', !open);
    nav.classList.toggle('open', !open);
  });
})();
</script>


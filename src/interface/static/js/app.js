/* MOTIE-ES: utilidades de interfaz */
document.addEventListener('DOMContentLoaded', () => {
  // Navegación por teclado: marcar enlace activo según ruta
  const path = window.location.pathname;
  document.querySelectorAll('.main-nav a').forEach(a => {
    if (a.getAttribute('href') === path) a.classList.add('active');
  });
});
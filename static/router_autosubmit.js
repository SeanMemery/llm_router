(() => {
  const forms = document.querySelectorAll("form .auto-submit");
  forms.forEach((input) => {
    input.addEventListener("change", (ev) => {
      const form = ev.target.closest("form");
      if (!form) return;
      form.submit();
    });
  });
})();

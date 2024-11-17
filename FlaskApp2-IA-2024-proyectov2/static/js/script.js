document.addEventListener("DOMContentLoaded", function () {
  const departureInput = document.getElementById("departure_time");
  if (!departureInput.value) {
    const now = new Date();
    now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
    departureInput.value = now.toISOString().slice(0, 16);
  }

  document.getElementById("routeForm").addEventListener("submit", function () {
    document.getElementById("loadingOverlay").style.display = "flex";
  });
});

let lastClickedInput = null;

document.getElementById("start").addEventListener("focus", function () {
  lastClickedInput = this;
});

document.getElementById("end").addEventListener("focus", function () {
  lastClickedInput = this;
});

function fillLocation(location) {
  if (lastClickedInput) {
    lastClickedInput.value = location;
  } else {
    document.getElementById("start").value = location;
  }
}

// Listen for the DOMContentLoaded event to ensure the DOM is fully loaded
document.addEventListener("DOMContentLoaded", function() {

  // Function to toggle menu items
  function toggleMenu(menuItems, className) {

    // Loop through each menu item
    menuItems.forEach(function(item) {

      // Create a span element to serve as the toggle button
      var toggleSpan = document.createElement('span');
      toggleSpan.className = 'toggle-btn ' + className;  // Set class for styling
      toggleSpan.innerHTML = "+";  // Initially set to "+"

      // Insert the toggle button before the first child of each menu item
      item.insertBefore(toggleSpan, item.firstChild);

      // Listen for clicks on the toggle button
      toggleSpan.addEventListener("click", function(event) {
        event.preventDefault();  // Prevent default hyperlink action

        // Get the next sibling element (assumed to be the submenu)
        var subMenu = item.nextElementSibling;

        // Toggle the submenu visibility
        if (subMenu.style.display === "none" || !subMenu.style.display) {
          subMenu.style.display = "block";  // Show
          toggleSpan.innerHTML = "-";  // Change to "-"
        } else {
          subMenu.style.display = "none";  // Hide
          toggleSpan.innerHTML = "+";  // Change to "+"
        }
      });
    });
  }

  // Get all level 1 menu items
  var menuItems1 = document.querySelectorAll(".toctree-l1 > a");
  toggleMenu(menuItems1, 'toggle-l1');

  // Get all level 2 menu items
  var menuItems2 = document.querySelectorAll(".toctree-l2 > a");
  toggleMenu(menuItems2, 'toggle-l2');

  // Listen for mouseover on level 1 items to make toggle buttons visible
  menuItems1.forEach(function(item) {
    item.addEventListener("mouseover", function(event) {
      item.querySelector('.toggle-l1').style.opacity = '1';  // Show level 1 toggle button
      var subItems = item.parentElement.querySelectorAll('.toggle-l2');  // Find all level 2 toggle buttons
      subItems.forEach(function(subItem) {
        subItem.style.opacity = '1';  // Show level 2 toggle buttons
      });
    });
  });
});

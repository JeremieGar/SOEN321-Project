document.getElementById("upload-form").addEventListener("submit", function(e) {
    e.preventDefault();

    // Show the loading spinner
    document.getElementById("spinner").style.display = "block";
    document.getElementById("uploadBtn").disabled = true; // Disable the button during upload

    // Simulate file submission and hide spinner after completion (you can update this with actual functionality later)
    setTimeout(function() {
        // Simulate successful file processing
        document.getElementById("spinner").style.display = "none";
        document.getElementById("uploadBtn").disabled = false; // Re-enable button
        alert("File successfully uploaded and processed!");
    }, 2000); // You can replace this with your file handling code later
});

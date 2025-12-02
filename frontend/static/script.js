document.getElementById("uploadForm")?.addEventListener("submit", async (e) => {
    e.preventDefault();

    const file = document.getElementById("file").files[0];
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/upload", {
        method: "POST",
        body: formData,
    });

    const data = await res.json();
    document.getElementById("response").textContent = JSON.stringify(data, null, 2);
});

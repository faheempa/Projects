const new_btn = document.querySelector(".new-btn");
const main = document.querySelector("main");

function savenote(e) {
    const notes = document.querySelectorAll(".note textarea");
    let data = []
    notes.forEach(note => {
        data.push(note.value);
    })
    localStorage.setItem("notes", JSON.stringify(data));
}

function addnote(e, text = "") {
    const note = document.createElement("div");
    note.classList.add("note");
    note.innerHTML = `
    <div class="note-head">
        <i class="fa fa-save"></i>
        <i class="fa fa-trash"></i>
    </div>
    <div class="note-body">
        <textarea>${text}</textarea>
    </div>`
    main.appendChild(note);
    note.querySelector(".fa-save").addEventListener("click", savenote);
    note.querySelector(".fa-trash").addEventListener("click", (e) => {
            note.remove();
            savenote();
        });
    note.addEventListener("mouseout", savenote);
    note.querySelector("textarea").focus();
}

new_btn.addEventListener("click", addnote);

(
    () => {
        const saved_notes = JSON.parse(localStorage.getItem("notes"));
        if (saved_notes.length !== 0) {
            saved_notes.forEach(note => {
                addnote("e", note);
            });
        }
        else {
            addnote();
        }
    }
)()
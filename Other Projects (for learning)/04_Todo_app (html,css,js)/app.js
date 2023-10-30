// declarations
const submit_btn = document.querySelector(".submit-btn")
const input = document.querySelector(".item-input")
const select = document.querySelector("select");


// functions
function localSaveItem(text) {
    let data;
    if (localStorage.getItem("items") === null)
        data = []
    else
        data = JSON.parse(localStorage.getItem("items"));

    data.push([text, 0]);
    localStorage.setItem("items", JSON.stringify(data));
}

function localRemoveItem(text) {
    let data = JSON.parse(localStorage.getItem("items"));
    let idx;
    data.every((value, index) => {
        if (value[0] === text) {
            idx = index;
            return false;
        }
        return true;
    })
    if (idx === undefined) return;
    data.splice(idx, 1);
    localStorage.setItem("items", JSON.stringify(data));
}
function localCheckItem(text) {
    let data = JSON.parse(localStorage.getItem("items"));
    data.every((value, index) => {
        if (value[0] === text) {
            value[1] = 1;
            return false;
        }
        return true;
    })
    localStorage.setItem("items", JSON.stringify(data));
}

function createItem(text) {
    const new_item = document.createElement("li");
    new_item.classList.add("item");
    new_item.innerHTML = `
    <div class="text">${text}</div>
    <i class="fa fa-check"></i>
    <i class="fa fa-trash"></i>`
    document.querySelector(".items").appendChild(new_item);

    new_item.querySelector(".fa-check").addEventListener("click", (e) => {
        new_item.classList.add("completed");
        localCheckItem(text);
    })

    new_item.querySelector(".fa-trash").addEventListener("click", (e) => {
        new_item.classList.add("remove-item");
        new_item.addEventListener("transitionend", () => {
            new_item.remove();
        })
        localRemoveItem(text);
    })
    localSaveItem(text);
}
function loadItem(text,completed) {
    const new_item = document.createElement("li");
    new_item.classList.add("item");
    new_item.innerHTML = `
    <div class="text">${text}</div>
    <i class="fa fa-check"></i>
    <i class="fa fa-trash"></i>`
    document.querySelector(".items").appendChild(new_item);

    new_item.querySelector(".fa-check").addEventListener("click", (e) => {
        new_item.classList.add("completed");
        localCheckItem(text);
    })

    new_item.querySelector(".fa-trash").addEventListener("click", (e) => {
        new_item.classList.add("remove-item");
        new_item.addEventListener("transitionend", () => {
            new_item.remove();
        })
        localRemoveItem(text);
    })
    if(completed) 
        new_item.classList.add("completed");
}

// events
submit_btn.addEventListener("click", (e) => {
    e.preventDefault();
    if (input.value === "") return;
    createItem(input.value);
    input.value = "";
})
select.addEventListener("click", () => {
    console.log(select.value);
    const items = document.querySelectorAll(".item");
    items.forEach(item => {
        let val = select.value;
        if (val === "not-completed" && item.classList.contains("completed") === true) {
            item.classList.add("hide");
        }
        else if (val === "completed" && item.classList.contains("completed") === false) {
            item.classList.add("hide");
        }
        else {
            item.classList.remove("hide");
        }
    })
})

// onload
function load() {
    if (localStorage.getItem("items") !== null) {
        const data = JSON.parse(localStorage.getItem("items"));
        data.forEach((value) => {
            if (value[1] == 1)
                loadItem(value[0], true);
            else
                loadItem(value[0], false);
        })
    }
}
load();


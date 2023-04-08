const buttons = document.querySelectorAll(".btn");
const clear_btn = document.querySelector(".btn-clear");
const equal_btn = document.querySelector(".btn-equal");
const screen = document.querySelector(".screen");

buttons.forEach(button => {
    button.addEventListener("click", function (e) {
        e.preventDefault();
        screen.value = screen.value + button.textContent;
    })
})

clear_btn.addEventListener("click", function (e) {
    screen.value = "";
})

equal_btn.addEventListener("click", function (e) {
    if (screen.value !== "") {
        try {
            let ans = eval(screen.value);
            screen.value = ans;
        } catch {
            screen.value = "Syntax Error";
            setTimeout(function () {
                screen.value = "";
            }, 500)
        }
    }
})

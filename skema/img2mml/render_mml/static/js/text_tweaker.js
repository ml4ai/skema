// Assumes eqn_src is available

/**
 * Showing different parts of the page
 */
const BASE_IMG_URL =
  "https://raw.githubusercontent.com/imzoc/mathpix-annotation/master/mathml-images/images_filtered/";

async function show_original_image(idx = 0) {
  let image_id = eqn_src[idx]["id"];
  $("#original-image-from-paper").attr("src", `${BASE_IMG_URL}${image_id}.png`);
}

async function show_image_info(idx = 0) {
  $("#img-count").text(`${idx + 1}/${eqn_src.length}`); // Image number

  // ID and confidence score
  let image_id_section = `<p><b>Image ID:</b> ${eqn_src[idx]["id"]}</p>`;
  let mathpix_confidence = (
    parseFloat(eqn_src[idx]["confidence"]) * 100
  ).toFixed(2);
  let mathpix_confidence_section = `<b> Mathpix Confidence Score:</b> ${mathpix_confidence}%`;
  $("#image-info").html(`${image_id_section}${mathpix_confidence_section}`);
}

async function show_latex(idx = 0) {
  let latex_code = eqn_src[idx]["latex"];
  $("#latex-code").text(latex_code); // code
  $("#latex-render").text(`$$${eqn_src[idx]["latex"]}$$`); // render
}

async function show_mathml(idx = 0) {
  let mathml_code = eqn_src[idx]["mathml"];
  $("#mathml-code").text(mathml_code); // code
  $("#mathml-render").html(mathml_code);
}

async function show_data(idx = 0) {
  show_latex(idx);
  show_mathml(idx);
  show_image_info(idx);
  show_original_image(idx);
  MathJax.typeset();
}

/**
 * Button Logic for changing the image and data currently being shown
 */
function update_btn_visibility() {
  $("#prev-btn").addClass("btn-secondary");
  $("#next-btn").addClass("btn-secondary");

  if (IDX > 0) {
    $("#prev-btn").addClass("btn-primary");
    $("#prev-btn").removeClass("btn-secondary");
  }
  if (IDX < eqn_src.length - 1) {
    $("#next-btn").addClass("btn-primary");
    $("#next-btn").removeClass("btn-secondary");
  }
}

$("#prev-btn").click(() => {
  if (IDX > 0) {
    IDX -= 1;
    show_data(IDX);
  }
  update_btn_visibility();
});

$("#next-btn").click(() => {
  if (IDX < eqn_src.length - 1) {
    IDX += 1;
    show_data(IDX);
  }
  update_btn_visibility();
});

/**
 * Preview changes being made to the code
 */
$("#latex-code").on("keyup", () => {
  let wip_latex_code = $("#latex-code").val();
  $("#latex-render").text(`$$${wip_latex_code}$$`);
  MathJax.typeset();
});

$("#mathml-code").on("keyup", () => {
  let wip_mathml_code = $("#mathml-code").val();
  $("#mathml-render").html(wip_mathml_code);
  MathJax.typeset();
});

/**
 * handle Image ID Search
 */
$("#image-id-search").submit(async (e) => {
  e.preventDefault();

  for (const [i, e] of eqn_src.entries()) {
    if (e["id"] === $("#image-id").val()) {
      IDX = i;
      show_data(IDX);
      update_btn_visibility();
      $("#image-id-search")[0].reset();
      break;
    }
  }
});

/**
 * Initial View
 */
var IDX = 0;
$(document).ready(function () {
  show_data(IDX);
  update_btn_visibility();
});

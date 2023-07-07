// Version of table generation script that uses jquery to construct table rows

const format_xml = require('xml-formatter');

function build_table() {
    let table = $("#table");
    let i = 0;

    for (let element of eqn_src) {

        let row = $("<tr/>", {class: "datum"});

        mml1 = `${element["mml1"]}`;

        mml1_formatted = format_xml(mml1, {
            indentation: '  ',
            collapseContent: true,
            lineSeparator: '\n'
        });

        var cell = $("<td>", {id: `mml1_src_${i}`})
            .append($("<div>", {class: 'pre'})
                .append($("<pre>").text(mml1_formatted)));
        row.append(cell);

        cell = $("<td/>", {id: `mml1_img_${i}`}).html(mml1);
        row.append(cell);

        if ("mml2" in element) {

            mml2 = `${element["mml2"]}`;

            mml2_formatted = format_xml(mml2, {
                indentation: '  ',
                collapseContent: true,
                lineSeparator: '\n'
            });

            cell = $("<td/>", {id: `mml2_img_${i}`}).html(mml2);
            row.append(cell);

            cell = $("<td>", {id: `mml2_src_${i}`})
                .append($("<div>", {class: 'pre'})
                    .append($("<pre>").text(mml2_formatted)));
            row.append(cell);

        } // end mml2 section

        table.append(row);

        i++;
    }
}


// triggers when HTML document is ready for processing
$(document).ready(function () {

    build_table();

});

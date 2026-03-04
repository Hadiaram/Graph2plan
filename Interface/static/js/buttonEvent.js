var focus_circle = false;
var focus_line = false;
var focus_rect = "";
var rect_type = false;
var Type = "";
var adjust_graph = false;
var createNewLine = false;
var isTrans = 0;
var islLoadTest = 0;
var selectRect;
var dragging_circle = null; // Track the currently dragging circle
var startRectvalue = [-1, -1, -1, -1];
var startPoint = [-1, -1, -1, -1, -1];
var RelRectvalue = [];
var rightDeleteMode = false; // Track delete mode for right-side interface
var deletedRightRooms = []; // Track deleted room IDs
var measureMode = false;       // Pixel ruler: active when true
var measurePoint1 = null;      // First clicked point {x, y} in SVG coords
$(document).ready(function () {
    start();//执行函数
    isTrans = 0;

});

function show(isShow) {
    // document.getElementById("rmlist").style.opacity = isShow;
    // document.getElementById("gooey-API").style.opacity = isShow;
    document.getElementById("leftbox").style.opacity = isShow;
    document.getElementById("rightbox").style.opacity = isShow;
    document.getElementById("listbox").style.opacity = isShow;
    document.getElementById("graphSearch").style.opacity = isShow;
    document.getElementById("Editing").style.opacity = isShow;

    document.getElementById("BedRoomVue").style.opacity = isShow;
    document.getElementById("BathRoomVue").style.opacity = isShow;
    document.getElementById("otherVue").style.opacity = isShow;
    document.getElementById("detailVue").style.opacity = isShow;
    document.getElementById("addVue").style.opacity = isShow;

}

$(document).ready(function () {
    show(0.0)
    setTimeout("show(1.0)", 12000)
    //load the start
    demo.init();
});

function start() {

    var leftsvg = document.getElementById('LeftGraphSVG');
    leftsvg.oncontextmenu = function () {
        return false;
    }

    $('#LeftGraphSVG').on('mousedown', function (e) {

        // Pixel ruler takes priority — don't place graph nodes while measuring
        if (measureMode) {
            // Use a child element's getScreenCTM() — child CTMs reliably include
            // the parent SVG's transform="scale(1.5)" attribute, whereas the root
            // SVG element's own getScreenCTM() may omit its own transform in some
            // browsers, causing a scale error.
            var layoutSVG = document.getElementById('LeftLayoutSVG');
            var ref = layoutSVG.querySelector('rect') || layoutSVG;
            var pt = layoutSVG.createSVGPoint();
            pt.x = e.clientX;
            pt.y = e.clientY;
            var svgCoords = pt.matrixTransform(ref.getScreenCTM().inverse());
            handleMeasureClick(svgCoords.x, svgCoords.y);
            return;
        }

        console.log("Left!");

        let selectX = e.clientX - leftsvg.getBoundingClientRect().left;
        let selectY = e.clientY - leftsvg.getBoundingClientRect().top;

        var roomSelect = -1;

        var arr, reg = new RegExp("(^| )ifSelectRoom=([^;]*)(;|$)");
        if (arr = document.cookie.match(reg)) {
            roomSelect = arr[2];
        } else {
            roomSelect = 0;
        }

        if (roomSelect == 1) {
            clearHighLight();

            var curRoom = "NULL";
            var curIndex = -1;

            arr, reg = new RegExp("(^| )RoomType=([^;]*)(;|$)");

            if (arr = document.cookie.match(reg)) {
                curRoom = arr[2];
            }

            arr, reg = new RegExp("(^| )CurNum=([^;]*)(;|$)");

            if (arr = document.cookie.match(reg)) {
                curIndex = arr[2];
            }
            var id = "TransCircle_" + curIndex + "_" + curRoom;
            // if (isTrans == 0) {
            //     document.getElementById("graphSearch").style = "display:flex;cursor: default;color: #000;text-align: center;vertical-align: middle;line-height: 26px;position: absolute;margin-left: 160px;"
            //
            // }
            CreateCircle(selectX / 2, selectY / 2, id);
            d3.select("body").select("#LeftGraphSVG").select("#" + id).attr('scalesize', 1);
            document.cookie = "ifSelectRoom=0";
            document.cookie = "RoomNum=" + (parseInt(curIndex) + 1)
        }
    })

    console.time('time');

    var model = 1;
    $.get("/index/Init/", {'start': model.toString()}, function () {
        console.log("load model success");
        console.timeEnd('time')

    })

    // Document-level handlers for smooth circle dragging
    $(document).on('mousemove', function(e) {
        circle_mousemove(e);
    });

    $(document).on('mouseup', function(e) {
        circle_mouseup();
    });

    animateHeight(true);
    animateHeight1(true);
    animateHeight2(true);
    animateHeight3(true);
    animateHeight4(true);

    // Delete Mode toggle for right-side interface
    $('#deleteMode').on('click', function() {
        rightDeleteMode = !rightDeleteMode;
        var btn = document.getElementById('deleteMode');
        if (rightDeleteMode) {
            btn.style.backgroundColor = '#43a047'; // Green when active
            btn.textContent = 'Delete Mode: ON';
            // Add visual feedback to right box
            document.getElementById('rightbox').style.border = '3px solid #e53935';
        } else {
            btn.style.backgroundColor = '#e53935'; // Red when inactive
            btn.textContent = 'Delete Mode: OFF';
            document.getElementById('rightbox').style.border = '2px solid #d2d2d2';
        }
    });

    // Reset button for right-side interface
    $('#resetRight').on('click', function() {
        console.log("Resetting right side view");
        resetRightSide();
        RightInit(); // Clear the SVGs

        // Get the current selected room ID from cookie to reload the view
        var arr, reg = new RegExp("(^| )roomID=([^;]*)(;|$)");
        var roomID = null;
        if (arr = document.cookie.match(reg)) {
            roomID = arr[2];
            if (roomID) {
                CreateRightImage(roomID); // Reload the image
            }
        }

        // Turn off delete mode
        if (rightDeleteMode) {
            $('#deleteMode').click();
        }
    });
}

function addLivingRoom(BtnID) {//这个加点的
    var arr, reg = new RegExp("(^| )RoomNum=([^;]*)(;|$)");
    var id = -1;
    if (arr = document.cookie.match(reg))
        id = parseInt(arr[2]);
    console.log(BtnID);
    var roomType = BtnID.split("_")[0];
    if (roomType == "BedRoom") {
        var Bedrandom = {0: "MasterRoom", 1: "SecondRoom", 2: "GuestRoom", 3: "ChildRoom", 4: "StudyRoom"};
        var rand = Math.random() * 5;

        roomType = Bedrandom[parseInt(rand)];
    }

    selectRoomType(roomType, id);
}

function clearHighLight() {
    var points = d3.select("body").select("#LeftGraphSVG").selectAll("circle").attr("stroke-width", 2);
}

function rect_clearHighLight() {
    var rects = d3.select("body").select("#LeftLayoutSVG").selectAll("rect").attr("stroke-width", 4);

}

function handleMeasureClick(x, y) {
    var svg = d3.select("#LeftLayoutSVG");

    if (!measurePoint1) {
        // First click — place a dot and wait for second click
        measurePoint1 = {x: x, y: y};

        // Clear any previous single-point marker
        svg.selectAll(".measureOverlay").remove();

        svg.append("circle")
            .attr("class", "measureOverlay")
            .attr("cx", x).attr("cy", y).attr("r", 2.5)
            .attr("fill", "#E65100")
            .attr("stroke", "white").attr("stroke-width", "0.5");

    } else {
        // Second click — draw line, dots, and distance label
        var x1 = measurePoint1.x, y1 = measurePoint1.y;
        var x2 = x, y2 = y;
        var dist = Math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
        var label = dist.toFixed(1) + " px";

        svg.selectAll(".measureOverlay").remove();

        // Line between the two points
        svg.append("line")
            .attr("class", "measureOverlay")
            .attr("x1", x1).attr("y1", y1)
            .attr("x2", x2).attr("y2", y2)
            .attr("stroke", "#E65100").attr("stroke-width", "1.5")
            .attr("stroke-dasharray", "4,2");

        // Endpoint dots
        [{x: x1, y: y1}, {x: x2, y: y2}].forEach(function(p) {
            svg.append("circle")
                .attr("class", "measureOverlay")
                .attr("cx", p.x).attr("cy", p.y).attr("r", 2.5)
                .attr("fill", "#E65100")
                .attr("stroke", "white").attr("stroke-width", "0.5");
        });

        // Distance label at midpoint
        var mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
        svg.append("text")
            .attr("class", "measureOverlay")
            .attr("x", mx).attr("y", my - 4)
            .attr("text-anchor", "middle")
            .attr("font-size", "8")
            .attr("font-weight", "bold")
            .attr("fill", "#E65100")
            .attr("stroke", "white").attr("stroke-width", "0.8")
            .attr("paint-order", "stroke")
            .text(label);

        console.log("[Measure] (" + x1.toFixed(1) + ", " + y1.toFixed(1) + ") → (" +
                    x2.toFixed(1) + ", " + y2.toFixed(1) + ") = " + label);

        // Reset so the next click starts a new measurement
        measurePoint1 = null;
    }
}

function selectRoomType(roomType, id) {
    document.cookie = "RoomType=" + roomType;
    document.cookie = "ifSelectRoom=1";
    document.cookie = "CurNum=" + id;
    // document.cookie = "CurNum=" + id.split("_")[1];
    var arr, reg = new RegExp("(^| )ifSelectRoom=([^;]*)(;|$)");
    if (arr = document.cookie.match(reg))
        console.log(arr[2]);
}

function init() {
    d3.select('body').select('#RightSVG').selectAll('line').remove();
    d3.select('body').select('#RightSVG').selectAll('circle').remove();

    d3.select('body').select('#RightLayoutSVG').selectAll('line').remove();
    d3.select('body').select('#RightLayoutSVG').selectAll('circle').remove();
    d3.select('body').select('#RightLayoutSVG').selectAll('rect').remove();
    d3.select('body').select('#RightLayoutSVG').selectAll('polygon').remove();
    d3.select('body').select('#RightLayoutSVG').selectAll('clipPath').remove();

    // d3.select('body').select('#LeftGraphSVG').selectAll('.TransLine').remove();
    // d3.select('body').select('#LeftGraphSVG').selectAll('.TransCircle').remove();
    document.getElementById("graphSearch").style = "cursor: default;color: #000;text-align: center;vertical-align: middle;line-height: 26px;position: absolute;margin-left: 360px;";

    d3.select('body').select('#LeftLayoutSVG').selectAll('rect').remove();
    d3.select('body').select('#LeftLayoutSVG').selectAll('polygon').remove();
    d3.select('body').select('#LeftLayoutSVG').selectAll('clipPath').remove();
    d3.select('body').select('#LeftLayoutSVG').selectAll('g').remove();

}

function RightInit() {
    d3.select('body').select('#RightSVG').selectAll('line').remove();
    d3.select('body').select('#RightSVG').selectAll('circle').remove();
    d3.select('body').select('#RightLayoutSVG').selectAll('line').remove();
    d3.select('body').select('#RightLayoutSVG').selectAll('circle').remove();
    d3.select('body').select('#RightLayoutSVG').selectAll('rect').remove();
    d3.select('body').select('#RightLayoutSVG').selectAll('polygon').remove();
    d3.select('body').select('#RightLayoutSVG').selectAll('clipPath').remove();

}

function ListBox(ret, rooms, metadata) {
    var roomList = ret;
    console.log("roomList" + roomList);
    console.log("metadata", metadata);
    var hsList = document.getElementById('hsList');
    while (hsList.hasChildNodes()) {
        hsList.removeChild(hsList.firstChild);
    }
    for (var i = roomList.length - 1; i >= 0; i--) {
        var hs = roomList[i];
        var itembt = document.createElement('button');

        // Create the main title with floor plan name
        var titleText = ret[i].split(".")[0];

        // Add match percentage if metadata is available
        if (metadata && metadata[i]) {
            var matchInfo = metadata[i];
            var matchPercent = matchInfo.match;
            var isFallback = matchInfo.fallback;

            // Create a styled match percentage badge
            var matchBadge = document.createElement('span');
            matchBadge.textContent = matchPercent + '%';
            matchBadge.style.cssText = 'float: right; background-color: ' +
                (isFallback ? '#FFA500' : '#4CAF50') + // Orange for fallback, green for exact match
                '; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px; margin-left: 5px;';

            itembt.innerHTML = titleText;
            itembt.appendChild(matchBadge);
        } else {
            itembt.innerHTML = titleText;
        }

        itembt.classList.add('api-title');
        itembt.classList.add('pngls');
        itembt.id = "Btn_" + ret[i];
        var itemimg = document.createElement('img');
        // itemimg.src="../static/Data/Img/52.png";
        //             itemimg.src="../static/Data/snapshot/"+ret[i];
        itemimg.src = "../static/Data/snapshot_train/" + ret[i];
        itembt.appendChild(itemimg);
        itembt.onclick = function () {
            RightInit();
            var all = document.getElementsByClassName("api-text");
            var i;
            for (i = 0; i < all.length; i++) {
                all[i].style.border = "0px";
            }
            d3.select('body').select('#LeftBaseSVG').selectAll('rect').remove();
            var parent = this.parentNode;
            parent.style.border = "2px solid #BEECFF";
            // d3.select('body').select('#LeftLayoutSVG').selectAll("svg > *").remove();
            console.time('time');
            console.log(this.id.split("_")[1]);
            var Rightid = this.id.split("_")[1];
            CreateRightImage(Rightid);
            // Store roomID in cookie for reset functionality
            document.cookie = "roomID=" + Rightid;
            document.getElementById("transfer").onclick = function () {
                d3.select('body').select('#LeftGraphSVG').selectAll('.TransLine').remove();
                d3.select('body').select('#LeftGraphSVG').selectAll('.TransCircle').remove();
                CreateLeftGraph(rooms, Rightid);
                // d3.select("body").select("#LeftGraphSVG").select("#" + roomid).attr('scalesize',1);
                document.getElementById("graphSearch").style = "display:none;cursor: default;color: #000;text-align: center;vertical-align: middle;line-height: 26px;position: absolute;margin-left: 160px;";
                isTrans = 1;
                document.getElementById("graphdiv").style = "display:block;cursor: default;color: #000;width: 90px;border: 2px solid #0072ca;border-radius: 30px;text-align: center;vertical-align: middle;line-height: 26px;height: 30px;position: absolute;margin-left: 300px;";
                document.getElementById("layoutdiv").style = "display:block;cursor: default;color: #000;width: 90px;border: 2px solid #0072ca;border-radius: 30px;text-align: center;vertical-align: middle;line-height: 26px;height: 30px;position: absolute;margin-left: 400px;";
            }
            console.timeEnd('time')
        }

        var itemdiv = document.createElement('div');
        itemdiv.classList.add('api-text');
        itemdiv.appendChild(itembt);

        var itemli = document.createElement('li');
        itemli.classList.add('col-sm-12');
        itemli.appendChild(itemdiv);
        hsList.insertBefore(itemli, hsList.firstChild);
    }
    console.time('time');
    // CreateRightImage(ret[0]);
// ocument.getElementById("transfer").onclick = function () {
//         CreateLeftGraph(rooms, ret[0]);}
    console.timeEnd('time')
}

function NumSearch() {
    document.getElementById("graphdiv").style = "display:none;cursor: default;color: #000;width: 90px;border: 2px solid #0072ca;border-radius: 30px;text-align: center;vertical-align: middle;line-height: 26px;height: 30px;position: absolute;margin-left: 300px;";
    document.getElementById("layoutdiv").style = "display:none;cursor: default;color: #000;width: 90px;border: 2px solid #0072ca;border-radius: 30px;text-align: center;vertical-align: middle;line-height: 26px;height: 30px;position: absolute;margin-left: 400px;";

    d3.select('body').select('#LeftGraphSVG').selectAll('.TransLine').remove();
    d3.select('body').select('#LeftGraphSVG').selectAll('.TransCircle').remove();
    document.cookie = "RoomNum=0";
    init();
    d3.select('body').select('#LeftBaseSVG').selectAll('rect').remove();
    d3.select("body").select("#LeftLayoutSVG").selectAll(".windowsline").remove();
    d3.select("body").selectAll(".UserPoint").attr("fill", "#6bdb6a").attr("stroke", 0);
    var hsname = null;
    var arr, reg = new RegExp("(^| )hsname=([^;]*)(;|$)");
    if (arr = document.cookie.match(reg))
        hsname = arr[2];

    var points = d3.select('body').select('#LeftGraphSVG').selectAll('circle');

    var rooms = [];
    rooms.push(hsname);
    var obj = Num();
    rooms.push(obj.roomactarr);
    rooms.push(obj.roomexaarr);
    rooms.push(obj.roomnumarr);

    points.each(function (d, i) {
        var room = [];
        room.push(this.id);
        room.push(this.cx.animVal.value);
        room.push(this.cy.animVal.value);
        rooms.push(room);
    });
    $.get("/index/NumSearch/", {'userInfo': JSON.stringify(rooms)}, function (ret) {
        // Handle new response format with backward compatibility
        var floorPlans = ret.floorPlans || ret;  // Use floorPlans if available, otherwise fall back to old format
        var metadata = ret.metadata || null;  // Get metadata if available
        ListBox(floorPlans, rooms, metadata);
    });
}


function roomcolor(rmcate) {
    switch (rmcate) {
        case "LivingRoom":
            var color = d3.rgb(244, 242, 229)
            break;
        case "MasterRoom":
            var color = d3.rgb(253, 244, 171)
            break;
        case "Kitchen":
            var color = d3.rgb(234, 216, 214)
            break;
        case "Bathroom":
            var color = d3.rgb(205, 233, 252);
            break;
        case "DiningRoom":
            var color = d3.rgb(244, 242, 229);
            break;
        case "ChildRoom":
            var color = d3.rgb(253, 244, 171);
            break;
        case "StudyRoom":
            var color = d3.rgb(253, 244, 171);
            break;
        case "SecondRoom":
            var color = d3.rgb(253, 244, 171);
            break;
        case "GuestRoom":
            var color = d3.rgb(253, 244, 171);
            break;
        case "Balcony":
            var color = d3.rgb(208, 216, 135);
            break;
        case "Entrance":
            var color = d3.rgb(244, 242, 229);
            break;
        case "Storage":
            var color = d3.rgb(249, 222, 189);
            break;
        case "Wall-in":
            var color = d3.rgb(202, 207, 239);
            break;
        case "External area":
            var color = d3.rgb(255, 255, 255);
            break;
        case "Exterior wall":
            var color = d3.rgb(79, 79, 79);
            break;
        case"Front door":
            var color = d3.rgb(255, 225, 25);
            break;
        case "Interior wall":
            var color = d3.rgb(128, 128, 128);
            break;
        case"Interior door":
            var color = d3.rgb(255, 255, 255);
            break;


        default:
            break
    }
    return color;
}

function CreateCircle(cx, cy, id, r) {
    if (r == undefined) {
        r = 5;
    }

    var title = id.split("_")[2];
    var circlecolor = roomcolor(title);
    d3.select('body').select('#LeftGraphSVG').append('circle')
        .attr("cx", cx)
        .attr("cy", cy)
        .attr("fill", circlecolor)
        .attr("r", r)
        .attr("stroke", "#000000")
        .attr("stroke-width", 2)
        .attr("id", id)
        .attr("class", "TransCircle")
        .on("mousedown", circle_mousedown)
        .on("dblclick", circle_dblclick)
        .append("title")//此处加入title标签
        .text(title);
}

function CreateLine(x1, y1, x2, y2, id) {
    d3.select('body').select('#LeftGraphSVG').append('line')
        .attr("x1", x1)
        .attr("y1", y1)
        .attr("x2", x2)
        .attr("y2", y2)
        .attr("stroke", "#000000")
        .attr("stroke-width", "2px")
        .attr("id", id)
        .attr("class", "TransLine")
        .on("mousedown", line_mousedown)
        .on("mouseup", line_mouseup)
}


function LoadTestBoundary(files) {
    init();
    if (islLoadTest == 1) {
        document.getElementById("BedRoomlb").innerHTML = "BedRoom";
        document.getElementById("BathRoomlb").innerHTML = "BathRoom";
        document.getElementById("otherlb").innerHTML = "Other Room Types";
        document.getElementById("detailedlb").innerHTML = "Detailed Bedroom Types";
        document.getElementById("graphdiv").style = "display:none;cursor: default;color: #000;width: 90px;border: 2px solid #0072ca;border-radius: 30px;text-align: center;vertical-align: middle;line-height: 26px;height: 30px;position: absolute;margin-left: 300px;";
        document.getElementById("layoutdiv").style = "display:none;cursor: default;color: #000;width: 90px;border: 2px solid #0072ca;border-radius: 30px;text-align: center;vertical-align: middle;line-height: 26px;height: 30px;position: absolute;margin-left: 400px;";

        // initVue();
    }
    d3.select('body').select('#LeftBaseSVG').selectAll("svg > *").remove();
    d3.select('body').select('#LeftGraphSVG').selectAll("svg > *").remove();
    d3.select('body').select('#LeftLayoutSVG').selectAll("svg > *").remove();
    d3.select('body').select('#RightLayoutSVG').selectAll("svg > *").remove();
    d3.select('body').select('#RightSVG').selectAll("svg > *").remove();
    document.getElementById('hsList').innerHTML = "";
    d3.select('body').select('#LeftBaseSVG').selectAll('polygon').remove();
    d3.select('body').select('#LeftBaseSVG').selectAll('line').remove();

    var file = files[0];
    console.log(file.name);
    document.cookie = "hsname=" + file.name;
    $.get("/index/LoadTestBoundary", {'testName': file.name}, function (ret) {
        var border = 4;
        islLoadTest = 1;
        var hsex = ret['exterior'];
        d3.select("#LeftBaseSVG")
            .append("polygon")
            .attr("points", hsex)
            .attr("fill", "none")
            .attr("stroke", roomcolor("Exterior wall"))
            .attr("stroke-width", border);
        var fontdoor_color = roomcolor("Front door");

        var door = ret['door'].split(",");
        d3.select('body').select('#LeftBaseSVG').append('line')
            .attr("x1", parseInt(door[0]))
            .attr("y1", door[1])
            .attr("x2", door[2])
            .attr("y2", door[3])
            .attr("stroke", fontdoor_color)
            .attr("stroke-width", border);

    })
    d3.select('body').select('#LeftBaseSVG').attr("transform", "scale(1.5)");
    d3.select('body').select('#LeftGraphSVG').attr("transform", "scale(1.5)");

    NumSearch();
}

function CreateLeftPlan(roombx, hsex, door, windows, indoor, windowsline, rmsize) {
    // Reset "Show Outside" toggle state whenever the floor plan is redrawn
    var outsideBtn = document.getElementById("showOutsideButton");
    if (outsideBtn) {
        outsideBtn.setAttribute("data-revealed", "false");
        outsideBtn.innerHTML = "👁 Show Outside";
        outsideBtn.style.backgroundColor = "#6A1B9A";
    }
    var expandBtn = document.getElementById("expandLivingRoom");
    if (expandBtn) {
        expandBtn.textContent = "Expand LR";
        expandBtn.style.backgroundColor = "#e65100";
    }
    var fillGapsBtn2 = document.getElementById("fillWallGapsButton");
    if (fillGapsBtn2) {
        fillGapsBtn2.textContent = "Fill Gaps";
        fillGapsBtn2.style.backgroundColor = "#388E3C";
    }

    d3.select('body').select('#LeftBaseSVG').selectAll('rect').remove();
    d3.select('body').select('#LeftLayoutSVG').selectAll("svg > *").remove();

    var interior_color = roomcolor("Interior wall");
    var border = 4;
    console.log("CreateLeftPlan", roombx);
    
    // Create clipPath first
    d3.select("#LeftLayoutSVG").append("clipPath")
        .attr("id", "clip-th")
        .append("polygon")
        .attr("points", hsex);
    
    for (var i = 0; i < roombx.length; i++) {
        var rx = roombx[i][0][0];
        var ry = roombx[i][0][1];
        var rw = roombx[i][0][2] - roombx[i][0][0];
        var rh = roombx[i][0][3] - roombx[i][0][1];
        var color = roomcolor(roombx[i][1][0]);
        var roomType = roombx[i][1][0];
        var tooltip = d3.select("body").append("div")
            .attr("class", "tooltip") //用于css设置类样式
            .attr("opacity", 0.0).attr("id", "tooltip" + roomType)
            .text(roomType);
        
        var rect = d3.select("#LeftLayoutSVG").append("rect").attr("x", rx)//每个矩形的起始x坐标
            .attr("y", ry)
            .attr("width", rw)
            .attr("height", rh)//每个矩形的高度
            .attr("stroke-width", border)//加边框厚度
            .attr("stroke", interior_color)
            .attr("fill", color)//填充颜色
            .attr("id", roomType + "_" + roombx[i][2])
            .on("mousedown", rect_mousedown)
            .on("mousemove", rect_mousemove)
            .on("mouseup", rect_mouseup)
            .on("click", rect_click)
            .on("dblclick", rect_dblclick);
        
        // Only apply clipping to non-balcony rooms
        if (roomType !== "Balcony") {
            rect.attr("clip-path", "url(#clip-th)");
        }
        
        rect.append("title")//此处加入title标签
            .text(roomType);//title标签的文字

    }

    d3.select("#LeftLayoutSVG")
        .append("polygon")
        .attr("points", hsex)
        .attr("fill", "none")
        .attr("stroke", roomcolor("Exterior wall"))
        .attr("stroke-width", border);
    var door = door.split(",");
    var fontdoor_color = roomcolor("Front door");
    d3.select('body').select('#LeftLayoutSVG').append('line')
        .attr("x1", parseInt(door[0]))
        .attr("y1", door[1])
        .attr("x2", door[2])
        .attr("y2", door[3])
        .attr("stroke", fontdoor_color)
        .attr("stroke-width", border);


    var wincolor = d3.rgb(195, 195, 195);
    // for (var i = 0; i < windows.length; i++) {
    //
    //     d3.select("#LeftBaseSVG").append("rect").attr("x", windows[i][0])//每个矩形的起始x坐标
    //         .attr("y", windows[i][1])
    //         .attr("width", windows[i][2])
    //         .attr("height", windows[i][3])//每个矩形的高度
    //         .attr("fill", "#ffffff")
    //         .attr("stroke",wincolor)
    //          .attr("stroke-width", 1);
    // }
//boudary clip
    //??
    // d3.select("body").select("#LeftCanvas").attr("style", "display:none");
    // for (var i = 0; i < windows.length; i++) {
    //
    //     d3.select("#LeftLayoutSVG").append("rect").attr("x", windows[i][0])//每个矩形的起始x坐标
    //         .attr("y", windows[i][1])
    //         .attr("width", windows[i][2])
    //         .attr("height", windows[i][3])//每个矩形的高度
    //         .attr("fill", wincolor).attr("fill","#ffffff" )
    //         .attr("stroke",wincolor)
    //          .attr("stroke-width", 1);
    // }
    // for (var i = 0; i < windowsline.length; i++) {
    //     d3.select('body').select('#LeftLayoutSVG').append('line')
    //         .attr("x1", windowsline[i][0])
    //         .attr("y1", windowsline[i][1])
    //         .attr("x2", windowsline[i][2])
    //         .attr("y2", windowsline[i][3]).attr("stroke",wincolor)
    //          .attr("stroke-width", 1) .attr("class", "windowsline");
    // }

    d3.select('body').select('#LeftLayoutSVG').attr("transform", "scale(1.5)");

}

function CreateRightImage(roomID) {
    $.getJSON("/index/LoadTrainHouse/", {'roomID': roomID}, function (ret) {
        // Store the data globally for proper node-to-box matching
        window.rightRoomData = {
            rmpos: ret['rmpos'],
            hsbox: ret['hsbox']
        };
        //Graph edge
        for (var i = 0; i < ret['hsedge'].length; i++) {
            var roomA = ret['hsedge'][i][0];
            var roomB = ret['hsedge'][i][1];

            d3.select('body').select('#RightSVG').append('line')
                .attr("x1", ret['rmpos'][roomA][2])
                .attr("y1", ret['rmpos'][roomA][3])
                .attr("x2", ret['rmpos'][roomB][2])
                .attr("y2", ret['rmpos'][roomB][3])
                .attr("stroke", "#000000")
                .attr("stroke-width", "2px")
                .attr("id", ret['rmpos'][roomA][1] + "-" + ret['rmpos'][roomB][1])
                .attr("class", "right-edge")
                .style("cursor", "pointer")
                .on("click", function() {
                    if (rightDeleteMode) {
                        removeRightEdge(this);
                    }
                })
                .on("mouseover", function() {
                    if (rightDeleteMode) {
                        d3.select(this)
                            .attr("stroke", "#e53935")
                            .attr("stroke-width", "4px");
                    }
                })
                .on("mouseout", function() {
                    if (rightDeleteMode) {
                        d3.select(this)
                            .attr("stroke", "#000000")
                            .attr("stroke-width", "2px");
                    }
                });
        }
        //Graph node size
        console.log(ret['rmsize']);
        console.log(ret['rmpos']);
        //Graph node
        for (var i = 0; i < ret['rmpos'].length; i++) {
            var nodeId = (i + 1) + "-" + ret['rmpos'][i][1];
            d3.select('body').select('#RightSVG').append('circle')
                .attr("cx", ret['rmpos'][i][2])
                .attr("cy", ret['rmpos'][i][3])
                .attr("fill", roomcolor(ret['rmpos'][i][1]))
                // .attr("r", 5)
                .attr("r", ret['rmsize'] [i][0][0])

                .attr("stroke", "#000000")
                .attr("stroke-width", 2)
                .attr("id", nodeId)
                .attr("class", "right-node")
                .attr("data-rmpos-index", i)  // Store rmpos index for matching
                .style("cursor", "pointer")
                .on("click", function() {
                    if (rightDeleteMode) {
                        removeRightNode(this);
                    }
                })
                .on("mouseover", function() {
                    if (rightDeleteMode) {
                        d3.select(this)
                            .attr("stroke", "#e53935")
                            .attr("stroke-width", 4);
                    }
                })
                .on("mouseout", function() {
                    if (rightDeleteMode) {
                        d3.select(this)
                            .attr("stroke", "#000000")
                            .attr("stroke-width", 2);
                    }
                });
        }
        d3.select('body').select('#RightSVG').attr("transform", "scale(1.5)");

        var border = 4;
        //Layout room
        var roombx = ret["hsbox"];
        var interiorwall_color = roomcolor("Interior wall");

        // Create clipPath first
        var hsex = ret["exterior"];
        d3.select("#RightLayoutSVG").append("clipPath")
            .attr("id", "Rightclip-th")
            .append("polygon")
            .attr("points", hsex);

        for (var i = 0; i < roombx.length; i++) {

            var rx = roombx[i][0][0];
            var ry = roombx[i][0][1];
            var rw = roombx[i][0][2] - roombx[i][0][0];
            var rh = roombx[i][0][3] - roombx[i][0][1];
            var color = roomcolor(roombx[i][1][0]);
            var roomType = roombx[i][1][0];

            // Apply clip-path to all rooms EXCEPT balconies
            var rect = d3.select("#RightLayoutSVG")
                .append("rect")
                .attr("x", rx)//每个矩形的起始x坐标
                .attr("y", ry)
                .attr("width", rw)
                .attr("height", rh)//每个矩形的高度
                .attr("stroke-width", 3)//加边框厚度
                .attr("stroke", interiorwall_color)
                .attr("fill", color)//填充颜色
                .attr("id", roomType)
                .attr("class", "right-room-" + i)
                .attr("data-hsbox-index", i)  // Store hsbox index for matching
                .style("cursor", "pointer")
                .on("click", function() {
                    if (rightDeleteMode) {
                        removeRightRoom(this);
                    }
                })
                .on("mouseover", function() {
                    if (rightDeleteMode) {
                        d3.select(this)
                            .attr("stroke", "#e53935")
                            .attr("stroke-width", 5);
                    }
                })
                .on("mouseout", function() {
                    if (rightDeleteMode) {
                        d3.select(this)
                            .attr("stroke", interiorwall_color)
                            .attr("stroke-width", 3);
                    }
                });

            // Only apply clipping to non-balcony rooms
            if (roomType !== "Balcony") {
                rect.attr("clip-path", "url(#Rightclip-th)");
            }
        }
        //Layout Boundary
        d3.select("#RightLayoutSVG")
            .append("polygon")
            .attr("points", hsex)
            .attr("fill", "none")
            .attr("stroke", roomcolor("Exterior wall"))
            .attr("stroke-width", 6);
        //door
        var door = ret['door'].split(",");

        var fontdoor_color = roomcolor("Front door");
        d3.select('body').select('#RightLayoutSVG').append('line')
            .attr("x1", door[0])
            .attr("y1", door[1])
            .attr("x2", door[2])
            .attr("y2", door[3])
            .attr("stroke", fontdoor_color)
            .attr("stroke-width", 6);
    });
    d3.select('body').select('#RightLayoutSVG').attr("transform", "scale(1.5)");

}

// Function to remove a graph node (circle) from the right side
function removeRightNode(nodeElement) {
    var nodeId = nodeElement.id;
    var rmposIndex = parseInt(nodeElement.getAttribute("data-rmpos-index"));
    console.log("Removing node: " + nodeId + ", rmpos index: " + rmposIndex);

    // Add to deleted list
    deletedRightRooms.push(nodeId);

    // Get node center coordinates from stored data
    if (window.rightRoomData && window.rightRoomData.rmpos[rmposIndex]) {
        var nodeCx = window.rightRoomData.rmpos[rmposIndex][2];
        var nodeCy = window.rightRoomData.rmpos[rmposIndex][3];
        var roomType = window.rightRoomData.rmpos[rmposIndex][1];

        console.log("Node center: (" + nodeCx + ", " + nodeCy + "), type: " + roomType);

        // Find ALL boxes that contain this node center, then pick the best match
        var matchingBoxes = [];
        d3.select('#RightLayoutSVG').selectAll('rect').each(function() {
            var boxIndex = parseInt(this.getAttribute("data-hsbox-index"));
            if (window.rightRoomData.hsbox[boxIndex]) {
                var boxData = window.rightRoomData.hsbox[boxIndex][0];
                var boxType = window.rightRoomData.hsbox[boxIndex][1][0];
                var x1 = boxData[0], y1 = boxData[1], x2 = boxData[2], y2 = boxData[3];

                // Check if node center is within box bounds (with small tolerance)
                var tolerance = 5;
                if (nodeCx >= x1 - tolerance && nodeCx <= x2 + tolerance &&
                    nodeCy >= y1 - tolerance && nodeCy <= y2 + tolerance) {
                    var area = (x2 - x1) * (y2 - y1);
                    var typeMatch = (boxType === roomType);
                    matchingBoxes.push({
                        element: this,
                        boxIndex: boxIndex,
                        area: area,
                        typeMatch: typeMatch
                    });
                }
            }
        });

        // If we found matching boxes, pick the best one
        if (matchingBoxes.length > 0) {
            // Sort by: 1) type match first, 2) then smallest area (most specific)
            matchingBoxes.sort(function(a, b) {
                if (a.typeMatch && !b.typeMatch) return -1;
                if (!a.typeMatch && b.typeMatch) return 1;
                return a.area - b.area; // Smaller area is better match
            });

            var bestMatch = matchingBoxes[0];
            console.log("Found best matching box at hsbox index: " + bestMatch.boxIndex +
                       " (type match: " + bestMatch.typeMatch + ", area: " + bestMatch.area + ")");

            // Remove only the best matching box
            d3.select(bestMatch.element)
                .transition()
                .duration(300)
                .attr("opacity", 0)
                .remove();
        }

        // Find and remove connected edges
        d3.select('#RightSVG').selectAll('line').each(function() {
            var lineId = this.id;
            if (lineId.includes(roomType)) {
                d3.select(this)
                    .transition()
                    .duration(300)
                    .attr("opacity", 0)
                    .remove();
            }
        });
    }

    // Remove the node with fade animation
    d3.select(nodeElement)
        .transition()
        .duration(300)
        .attr("opacity", 0)
        .remove();
}

// Function to remove a room box (rectangle) from the right side
function removeRightRoom(roomElement) {
    var roomId = roomElement.id; // This is the room type (e.g., "Bedroom")
    var hsboxIndex = parseInt(roomElement.getAttribute("data-hsbox-index"));
    console.log("Removing room box: " + roomId + ", hsbox index: " + hsboxIndex);

    // Get box bounds from stored data
    if (window.rightRoomData && window.rightRoomData.hsbox[hsboxIndex]) {
        var boxData = window.rightRoomData.hsbox[hsboxIndex][0];
        var boxType = window.rightRoomData.hsbox[hsboxIndex][1][0];
        var x1 = boxData[0], y1 = boxData[1], x2 = boxData[2], y2 = boxData[3];
        var boxCx = (x1 + x2) / 2;
        var boxCy = (y1 + y2) / 2;

        console.log("Box bounds: (" + x1 + ", " + y1 + ") to (" + x2 + ", " + y2 + "), type: " + boxType);

        // Find ALL nodes within this box, then pick the best match
        var matchingNodes = [];
        d3.select('#RightSVG').selectAll('circle').each(function() {
            var rmposIndex = parseInt(this.getAttribute("data-rmpos-index"));
            if (window.rightRoomData.rmpos[rmposIndex]) {
                var nodeCx = window.rightRoomData.rmpos[rmposIndex][2];
                var nodeCy = window.rightRoomData.rmpos[rmposIndex][3];
                var nodeType = window.rightRoomData.rmpos[rmposIndex][1];

                // Check if node center is within box bounds (with small tolerance)
                var tolerance = 5;
                if (nodeCx >= x1 - tolerance && nodeCx <= x2 + tolerance &&
                    nodeCy >= y1 - tolerance && nodeCy <= y2 + tolerance) {
                    // Calculate distance from node to box center
                    var distSq = (nodeCx - boxCx) * (nodeCx - boxCx) + (nodeCy - boxCy) * (nodeCy - boxCy);
                    var typeMatch = (nodeType === boxType);
                    matchingNodes.push({
                        element: this,
                        rmposIndex: rmposIndex,
                        nodeType: nodeType,
                        distSq: distSq,
                        typeMatch: typeMatch
                    });
                }
            }
        });

        // If we found matching nodes, pick the best one
        if (matchingNodes.length > 0) {
            // Sort by: 1) type match first, 2) then closest to box center
            matchingNodes.sort(function(a, b) {
                if (a.typeMatch && !b.typeMatch) return -1;
                if (!a.typeMatch && b.typeMatch) return 1;
                return a.distSq - b.distSq; // Closer is better
            });

            var bestMatch = matchingNodes[0];
            console.log("Found best matching node at rmpos index: " + bestMatch.rmposIndex +
                       ", type: " + bestMatch.nodeType +
                       " (type match: " + bestMatch.typeMatch + ")");

            // Remove only the best matching node
            d3.select(bestMatch.element)
                .transition()
                .duration(300)
                .attr("opacity", 0)
                .remove();

            // Remove connected edges
            d3.select('#RightSVG').selectAll('line').each(function() {
                var lineId = this.id;
                if (lineId.includes(bestMatch.nodeType)) {
                    d3.select(this)
                        .transition()
                        .duration(300)
                        .attr("opacity", 0)
                        .remove();
                }
            });
        }
    }

    // Remove the rectangle with fade animation
    d3.select(roomElement)
        .transition()
        .duration(300)
        .attr("opacity", 0)
        .remove();
}

// Function to remove a graph edge (line) from the right side
function removeRightEdge(edgeElement) {
    var edgeId = edgeElement.id;
    console.log("Removing edge: " + edgeId);

    // Remove the edge with fade animation
    d3.select(edgeElement)
        .transition()
        .duration(300)
        .attr("opacity", 0)
        .remove();
}

// Function to clear all deleted rooms and reset the right side
function resetRightSide() {
    deletedRightRooms = [];
    console.log("Reset right side - cleared deleted rooms list");
}

function GetEditGraph(ret) {
    var hsname = null;
    var arr, reg = new RegExp("(^| )hsname=([^;]*)(;|$)");
    if (arr = document.cookie.match(reg))
        hsname = arr[2];

    var newCircles = d3.select("body").select("#LeftGraphSVG").selectAll("circle");
    console.log(newCircles);
    var GraphNode = [];
    newCircles.each(function (d, i) {
        // console.log(this.cx.animVal.value, this.cy.animVal.value, this.id);
        var newnode = [];
        var idlist = this.id.split("_");
        newnode.push(idlist[1]);
        newnode.push(idlist[2]);
        newnode.push(this.cx.animVal.value);
        newnode.push(this.cy.animVal.value);
        console.log(this.attributes.scalesize.value);
        newnode.push(this.attributes.scalesize.value);
        GraphNode.push(newnode);
        // GraphNode.push(newnode);
    });
    var newLine = d3.select("body").select("#LeftGraphSVG").selectAll("line");
    // console.log(newLine);
    var GraphEdge = [];
    newLine.each(function (d, i) {
        var newedge = [];
        var idlist = this.id.split("_");
        newedge.push(idlist[1]);
        newedge.push(idlist[2]);
        GraphEdge.push(newedge);
    });
    var NewGraph = [];
    NewGraph.push(GraphNode);
    NewGraph.push(GraphEdge);
    if (ret != 0) {
        NewGraph.push(ret);
    }
    return NewGraph
}

function GetEditLayout() {
    var hsname = null;
    var arr, reg = new RegExp("(^| )hsname=([^;]*)(;|$)");
    if (arr = document.cookie.match(reg))
        hsname = arr[2];

    var newRects = d3.select("body").select("#LeftLayoutSVG").selectAll("rect");
    console.log("newRects", newRects);
    var LayRect = [];
    newRects.each(function (d, i) {
        var newrect = [];
        var idlist = this.id.split("_");
        newrect.push(idlist[0]);
        newrect.push(idlist[1]);
        newrect.push(this.x.animVal.value);
        newrect.push(this.y.animVal.value);
        newrect.push(this.x.animVal.value + this.width.animVal.value);
        newrect.push(this.y.animVal.value + this.height.animVal.value);
        LayRect.push(newrect);
    });
    return LayRect
}

function GraphSearch() {
    document.getElementById("graphdiv").style = "display:none;cursor: default;color: #000;width: 90px;border: 2px solid #0072ca;border-radius: 30px;text-align: center;vertical-align: middle;line-height: 26px;height: 30px;position: absolute;margin-left: 300px;";
    document.getElementById("layoutdiv").style = "display:none;cursor: default;color: #000;width: 90px;border: 2px solid #0072ca;border-radius: 30px;text-align: center;vertical-align: middle;line-height: 26px;height: 30px;position: absolute;margin-left: 400px;";

    var hsname = null;
    var arr, reg = new RegExp("(^| )hsname=([^;]*)(;|$)");
    if (arr = document.cookie.match(reg))
        hsname = arr[2];
    NewGraph = GetEditGraph(0);
    var rooms = [];
    rooms.push(hsname);
    var obj = Num();
    var Numrooms = [];
    Numrooms.push(obj.roomactarr);
    Numrooms.push(obj.roomexaarr);
    Numrooms.push(obj.roomnumarr);
    $.get("/index/GraphSearch/", {
        'NewGraph': JSON.stringify(NewGraph),
        'userRoomID': hsname,
        'Numrooms': JSON.stringify(Numrooms),
    }, function (ret) {
        // Handle new response format with backward compatibility
        var floorPlans = ret.floorPlans || ret;  // Use floorPlans if available, otherwise fall back to old format
        var metadata = ret.metadata || null;  // Get metadata if available
        ListBox(floorPlans, rooms, metadata)
    });
}

function CreateLeftFloorPlan(boxes, exterior, door) {
    // Reset "Show Outside" toggle state whenever the floor plan is redrawn
    var outsideBtn = document.getElementById("showOutsideButton");
    if (outsideBtn) {
        outsideBtn.setAttribute("data-revealed", "false");
        outsideBtn.innerHTML = "👁 Show Outside";
        outsideBtn.style.backgroundColor = "#6A1B9A";
    }
    var expandBtn = document.getElementById("expandLivingRoom");
    if (expandBtn) {
        expandBtn.textContent = "Expand LR";
        expandBtn.style.backgroundColor = "#e65100";
    }
    var fillGapsBtn2 = document.getElementById("fillWallGapsButton");
    if (fillGapsBtn2) {
        fillGapsBtn2.textContent = "Fill Gaps";
        fillGapsBtn2.style.backgroundColor = "#388E3C";
    }

    // Clear existing floor plan
    d3.select('#LeftLayoutSVG').selectAll('rect').remove();
    d3.select('#LeftLayoutSVG').selectAll('polygon').remove();
    d3.select('#LeftLayoutSVG').selectAll('line').remove();
    d3.select('#LeftLayoutSVG').selectAll('clipPath').remove();

    var border = 4;
    var interiorwall_color = roomcolor("Interior wall");

    // Create clipPath for boundary
    d3.select("#LeftLayoutSVG").append("clipPath")
        .attr("id", "left-clip-transferred")
        .append("polygon")
        .attr("points", exterior);

    // Draw room rectangles
    for (var i = 0; i < boxes.length; i++) {
        var rx = boxes[i][0][0];
        var ry = boxes[i][0][1];
        var rw = boxes[i][0][2] - boxes[i][0][0];
        var rh = boxes[i][0][3] - boxes[i][0][1];
        var roomType = boxes[i][1][0];
        var color = roomcolor(roomType);

        var rect = d3.select("#LeftLayoutSVG")
            .append("rect")
            .attr("x", rx)
            .attr("y", ry)
            .attr("width", rw)
            .attr("height", rh)
            .attr("stroke-width", border)
            .attr("stroke", interiorwall_color)
            .attr("fill", color)
            .attr("id", "transferred_" + roomType + "_" + i);

        // Apply clipping (except balconies)
        if (roomType !== "Balcony") {
            rect.attr("clip-path", "url(#left-clip-transferred)");
        }
    }

    // Draw boundary polygon
    d3.select("#LeftLayoutSVG")
        .append("polygon")
        .attr("points", exterior)
        .attr("fill", "none")
        .attr("stroke", roomcolor("Exterior wall"))
        .attr("stroke-width", border);

    // Draw door
    var doorCoords = door.split(",");
    d3.select('#LeftLayoutSVG').append('line')
        .attr("x1", doorCoords[0])
        .attr("y1", doorCoords[1])
        .attr("x2", doorCoords[2])
        .attr("y2", doorCoords[3])
        .attr("stroke", roomcolor("Front door"))
        .attr("stroke-width", border);

    d3.select('#LeftLayoutSVG').attr("transform", "scale(1.5)");
}

function CreateLeftGraph(rooms, roomID) {
    $.getJSON("/index/TransGraph/", {'userInfo': rooms.toString(), 'roomID': roomID}, function (ret) {
        //     $.getJSON("/index/TransGraph_net/", {'userInfo': rooms.toString(), 'roomID': roomID}, function (ret) {
        // Show Auto-Adjust button when graph is transferred
        document.getElementById("AutoAdjust").style.display = "block";

        // Show Floor Plan visualization button
        document.getElementById("ShowFloorPlan").style.display = "block";
        document.getElementById("ShowFloorPlan").onclick = function () {
            console.log("Showing current floor plan...");

            // Get current graph state (edited nodes/edges)
            var currentGraph = GetEditGraph(ret['rmpos']);

            console.log("Current graph to send:", currentGraph);
            console.log("userRoomID:", rooms.toString().split(',')[0]);
            console.log("adptRoomID:", roomID);

            // Send to backend to regenerate floor plan based on current graph using AI model
            console.log("\n" + "=".repeat(80));
            console.log("🚀 [FRONTEND] Show Floor Plan - Sending AdjustGraph request");
            console.log("=".repeat(80));
            console.log("📤 Request parameters:");
            console.log("   → userRoomID:", rooms.toString().split(',')[0]);
            console.log("   → adptRoomID:", roomID);
            console.log("   → NewGraph:", currentGraph);
            console.log("   → NewGraph structure: nodes=" + (currentGraph[0] ? currentGraph[0].length : 0) + 
                        ", edges=" + (currentGraph[1] ? currentGraph[1].length : 0) +
                        ", oldNodes=" + (currentGraph[2] ? currentGraph[2].length : 0));
            
            var requestStartTime = performance.now();
            $.get("/index/AdjustGraph/", {
                'NewGraph': JSON.stringify(currentGraph),
                'userRoomID': rooms.toString().split(',')[0],
                'adptRoomID': roomID
            }, function (adjust_ret) {
                var requestEndTime = performance.now();
                console.log("\n✅ [FRONTEND] AdjustGraph response received");
                console.log("   ⏱️  Request time: " + (requestEndTime - requestStartTime).toFixed(2) + "ms");
                console.log("📥 Response data:");
                console.log("   → roomret entries:", adjust_ret['roomret'] ? adjust_ret['roomret'].length : 0);
                console.log("   → hsedge entries:", adjust_ret['hsedge'] ? adjust_ret['hsedge'].length : 0);
                console.log("   → exterior:", adjust_ret['exterior'] ? "present" : "missing");
                console.log("   → door:", adjust_ret['door'] ? "present" : "missing");
                console.log("   → Full response:", adjust_ret);

                // Use the AI-generated boxes from current graph
                console.log("🎨 [FRONTEND] Rendering floor plan with CreateLeftFloorPlan...");
                CreateLeftFloorPlan(adjust_ret['roomret'], adjust_ret['exterior'], adjust_ret['door']);
                console.log("✅ [FRONTEND] Floor plan rendering completed");
                console.log("=".repeat(80) + "\n");
            }).fail(function(xhr, status, error) {
                var requestEndTime = performance.now();
                console.error("\n" + "=".repeat(80));
                console.error("❌ [FRONTEND] AdjustGraph request FAILED!");
                console.error("=".repeat(80));
                console.error("   ⏱️  Request time: " + (requestEndTime - requestStartTime).toFixed(2) + "ms");
                console.error("   Status:", status);
                console.error("   Error:", error);
                console.error("   HTTP Status:", xhr.status);
                console.error("   Response Text:", xhr.responseText);
                console.error("   Response Headers:", xhr.getAllResponseHeaders());
                console.error("=".repeat(80) + "\n");
                alert("Error regenerating floor plan. Check browser console for details.");
            });
        };

        document.getElementById("Generate").onclick = function () {
            var AdjustNewGraph = [];
            AdjustNewGraph = GetEditGraph(ret['rmpos']);
            // NewGraph.push(ret['rmpos']);

            console.log("\n" + "=".repeat(80));
            console.log("🚀 [FRONTEND] Generate - Sending AdjustGraph request");
            console.log("=".repeat(80));
            console.log("📤 Request parameters:");
            console.log("   → userRoomID:", rooms.toString().split(',')[0]);
            console.log("   → adptRoomID:", roomID);
            console.log("   → AdjustNewGraph:", AdjustNewGraph);
            
            var requestStartTime = performance.now();
            $.get("/index/AdjustGraph/", {
                'NewGraph': JSON.stringify(AdjustNewGraph),
                'userRoomID': rooms.toString().split(',')[0],
                'adptRoomID': roomID
            }, function (adjust_ret) {
                var requestEndTime = performance.now();
                console.log("\n✅ [FRONTEND] Generate response received");
                console.log("   ⏱️  Request time: " + (requestEndTime - requestStartTime).toFixed(2) + "ms");
                console.log("📥 Response data:", adjust_ret);
                console.log("   → rmpos:", adjust_ret['rmpos']);
                
                // console.log("ret");
                console.log("🎨 [FRONTEND] Rendering with CreateLeftPlan...");
                CreateLeftPlan(adjust_ret['roomret'], adjust_ret['exterior'], adjust_ret["door"], adjust_ret["windows"], adjust_ret["indoor"], adjust_ret["windowsline"]);
                d3.select('body').select('#LeftGraphSVG').selectAll('circle').attr("r", 0);
                document.getElementById("OptimizeLayout").style.display = "block";
                var showOutsideBtn = document.getElementById("showOutsideButton");
                if (showOutsideBtn) { showOutsideBtn.style.display = "block"; }
                var expandLRBtn = document.getElementById("expandLivingRoom");
                if (expandLRBtn) { expandLRBtn.style.display = "block"; }
                var dxfBtn = document.getElementById("exportDXFButton");
                if (dxfBtn) { dxfBtn.style.display = "block"; }
                var fillGapsBtn = document.getElementById("fillWallGapsButton");
                if (fillGapsBtn) { fillGapsBtn.style.display = "block"; }
                var snapRoomsBtn = document.getElementById("snapRoomsButton");
                if (snapRoomsBtn) { snapRoomsBtn.style.display = "block"; }
                var fillLRBtn = document.getElementById("fillLivingRoomButton");
                if (fillLRBtn) { fillLRBtn.style.display = "block"; }
                var logBndBtn = document.getElementById("logBoundariesButton");
                if (logBndBtn) { logBndBtn.style.display = "block"; }
                var measureBtn = document.getElementById("measureButton");
                if (measureBtn) { measureBtn.style.display = "block"; }
                console.log(adjust_ret['rmpos']);

                for (var i = 0; i < adjust_ret['rmpos'].length; i++) {
                    var id = null;
                    var Circlesize = null;
                    id = "TransCircle" + "_" + adjust_ret['rmpos'][i][4] + "_" + adjust_ret['rmpos'][i][1];
                    Circlesize = d3.select("body").select("#LeftGraphSVG").select("#" + id);
                    // console.log(id);
                    // console.log(adjust_ret['rmsize'][i][0]);
                    if (parseInt((adjust_ret['rmsize'][i][0])) == 0) {
                        adjust_ret['rmsize'][i][0] = 4;
                    }
                    Circlesize.attr("r", adjust_ret['rmsize'][i][0]);
                }
                console.log("✅ [FRONTEND] Generate completed");
                console.log("=".repeat(80) + "\n");
            }).fail(function(xhr, status, error) {
                var requestEndTime = performance.now();
                console.error("\n" + "=".repeat(80));
                console.error("❌ [FRONTEND] Generate request FAILED!");
                console.error("=".repeat(80));
                console.error("   ⏱️  Request time: " + (requestEndTime - requestStartTime).toFixed(2) + "ms");
                console.error("   Status:", status);
                console.error("   Error:", error);
                console.error("   Response:", xhr.responseText);
                console.error("=".repeat(80) + "\n");
                alert("Error generating floor plan. Check browser console for details.");
            });
        };

        // Optimize button handler
        document.getElementById("OptimizeLayout").onclick = function () {
            console.log("🔧 [FRONTEND] Optimize — sending OptimizeLayout request");
            var btn = document.getElementById("OptimizeLayout");
            btn.textContent = "Optimizing…";
            btn.style.backgroundColor = "#616161";

            $.get("/index/OptimizeLayout/", {}, function (opt_ret) {
                console.log("✅ [FRONTEND] OptimizeLayout response received");
                CreateLeftPlan(opt_ret['roomret'], opt_ret['exterior'], opt_ret["door"],
                               opt_ret["windows"], opt_ret["indoor"], opt_ret["windowsline"]);
                btn.textContent = "Optimize";
                btn.style.backgroundColor = "#00897b";
            }).fail(function (xhr, status, error) {
                console.error("❌ [FRONTEND] OptimizeLayout FAILED:", xhr.responseText);
                alert("Optimization failed. Check the server console for details.");
                btn.textContent = "Optimize";
                btn.style.backgroundColor = "#00897b";
            });
        };

        // Expand Living Room button handler
        document.getElementById("expandLivingRoom").onclick = function () {
            var btn = document.getElementById("expandLivingRoom");
            btn.textContent = "Expanding…";
            btn.style.backgroundColor = "#616161";

            $.get("/index/ExpandLivingRoom/", {}, function (ret) {
                console.log("✅ [FRONTEND] ExpandLivingRoom response received");
                CreateLeftPlan(ret['roomret'], ret['exterior'], ret["door"],
                               ret["windows"], ret["indoor"], ret["windowsline"]);
                btn.textContent = "Expand LR";
                btn.style.backgroundColor = "#e65100";
            }).fail(function (xhr, status, error) {
                console.error("❌ [FRONTEND] ExpandLivingRoom FAILED:", xhr.responseText);
                alert("Expansion failed. Check the server console for details.");
                btn.textContent = "Expand LR";
                btn.style.backgroundColor = "#e65100";
            });
        };

        // Fill Wall Gaps button handler
        document.getElementById("fillWallGapsButton").onclick = function () {
            var btn = document.getElementById("fillWallGapsButton");
            btn.textContent = "Filling…";
            btn.style.backgroundColor = "#616161";

            $.get("/index/FillWallGaps/", {}, function (ret) {
                console.log("✅ [FRONTEND] FillWallGaps response received");
                CreateLeftPlan(ret['roomret'], ret['exterior'], ret["door"],
                               ret["windows"], ret["indoor"], ret["windowsline"]);
                btn.textContent = "Fill Gaps";
                btn.style.backgroundColor = "#388E3C";
            }).fail(function (xhr, status, error) {
                console.error("❌ [FRONTEND] FillWallGaps FAILED:", xhr.responseText);
                alert("Fill gaps failed. Check the server console for details.");
                btn.textContent = "Fill Gaps";
                btn.style.backgroundColor = "#388E3C";
            });
        };

        // Snap Rooms button handler
        document.getElementById("snapRoomsButton").onclick = function () {
            var btn = document.getElementById("snapRoomsButton");
            btn.textContent = "Snapping…";
            btn.style.backgroundColor = "#616161";

            $.get("/index/SnapRooms/", {}, function (ret) {
                console.log("✅ [FRONTEND] SnapRooms response received");
                CreateLeftPlan(ret['roomret'], ret['exterior'], ret["door"],
                               ret["windows"], ret["indoor"], ret["windowsline"]);
                btn.textContent = "Snap Rooms";
                btn.style.backgroundColor = "#6A1B9A";
            }).fail(function (xhr, status, error) {
                console.error("❌ [FRONTEND] SnapRooms FAILED:", xhr.responseText);
                alert("Snap rooms failed. Check the server console for details.");
                btn.textContent = "Snap Rooms";
                btn.style.backgroundColor = "#6A1B9A";
            });
        };

        // Fill Living Room button handler
        document.getElementById("fillLivingRoomButton").onclick = function () {
            var btn = document.getElementById("fillLivingRoomButton");
            btn.textContent = "Filling…";
            btn.style.backgroundColor = "#616161";

            $.get("/index/FillLivingRoom/", {}, function (ret) {
                console.log("✅ [FRONTEND] FillLivingRoom response received");
                CreateLeftPlan(ret['roomret'], ret['exterior'], ret["door"],
                               ret["windows"], ret["indoor"], ret["windowsline"]);
                btn.textContent = "Fill LR";
                btn.style.backgroundColor = "#BF360C";
            }).fail(function (xhr, status, error) {
                console.error("❌ [FRONTEND] FillLivingRoom FAILED:", xhr.responseText);
                alert("Fill LR failed. Check the server console for details.");
                btn.textContent = "Fill LR";
                btn.style.backgroundColor = "#BF360C";
            });
        };

        // Pixel ruler: Measure button toggle handler
        document.getElementById("measureButton").onclick = function () {
            measureMode = !measureMode;
            measurePoint1 = null;

            var btn = document.getElementById("measureButton");
            var svgEl = document.getElementById('LeftGraphSVG');

            if (measureMode) {
                btn.style.backgroundColor = "#E65100";
                btn.textContent = "📏 Measuring…";
                svgEl.style.cursor = "crosshair";
            } else {
                btn.style.backgroundColor = "#795548";
                btn.textContent = "📏 Measure";
                svgEl.style.cursor = "";
                d3.select("#LeftLayoutSVG").selectAll(".measureOverlay").remove();
            }
        };

        // Log Boundary button handler
        document.getElementById("logBoundariesButton").onclick = function () {
            var arr, reg = new RegExp("(^| )hsname=([^;]*)(;|$)");
            var hsname;
            if (arr = document.cookie.match(reg))
                hsname = arr[2];

            if (!hsname) {
                alert("Please load a boundary file first!");
                return;
            }

            var userRoomID = hsname.split(".")[0];

            var logBtn = document.getElementById("logBoundariesButton");
            var originalText = logBtn.innerHTML;
            logBtn.innerHTML = "⏳ Loading...";
            logBtn.style.backgroundColor = "#757575";
            logBtn.style.cursor = "wait";

            console.log("[Log Boundary] Requesting boundary log for:", userRoomID);

            $.get("/index/Log_Boundaries/", { 'userRoomID': userRoomID }, function (data) {
                logBtn.innerHTML = originalText;
                logBtn.style.backgroundColor = "#00897B";
                logBtn.style.cursor = "pointer";

                if (data.success) {
                    // Print full detail to browser console
                    console.log("[Log Boundary] ============================================================");
                    console.log("[Log Boundary] Floor plan:", data.floor_plan_id);
                    console.log("[Log Boundary] Boundary extents:", data.boundary_extents);
                    console.log("[Log Boundary] Boundary points (" + data.boundary_point_count + "):");
                    data.boundary_points.forEach(function(pt) {
                        var line = "  pt[" + pt.index + "]  x=" + pt.x + "  y=" + pt.y;
                        if (pt.direction) line += "  dir=" + pt.direction;
                        if (pt.is_new !== undefined) line += "  isNew=" + pt.is_new;
                        console.log("[Log Boundary]" + line);
                    });
                    console.log("[Log Boundary] Wall segments (" + data.wall_segments.length + "):");
                    data.wall_segments.forEach(function(w) {
                        var line = "  wall[" + w.index + "]  (" + w.x1 + "," + w.y1 + ") → (" + w.x2 + "," + w.y2 + ")  len=" + w.length;
                        if (w.direction) line += "  dir=" + w.direction;
                        console.log("[Log Boundary]" + line);
                    });
                    console.log("[Log Boundary] Rooms (" + data.room_count + "):");
                    data.rooms.forEach(function(r) {
                        var line = "  [" + r.index + "] " + r.type_name +
                                   "  (" + r.x1 + "," + r.y1 + ")-(" + r.x2 + "," + r.y2 + ")" +
                                   "  closest_wall=" + r.closest_wall_name +
                                   "  wall_dist=" + r.closest_wall_dist;
                        if (r.escape_area_px2 !== undefined)
                            line += "  escape=" + r.escape_area_px2 + "px²";
                        if (!r.inside_boundary_extents)
                            line += "  ⚠ OUTSIDE extents";
                        console.log("[Log Boundary]" + line);
                    });
                    console.log("[Log Boundary] ============================================================");

                    // SVG overlay: wall labels + room-to-nearest-wall dashed lines
                    d3.select("#LeftBaseSVG").selectAll(".logBndOverlay").remove();
                    d3.select("#LeftLayoutSVG").selectAll(".logBndOverlay").remove();

                    // Label each wall at its midpoint
                    data.wall_segments.forEach(function(w) {
                        var mx = (w.x1 + w.x2) / 2;
                        var my = (w.y1 + w.y2) / 2;
                        d3.select("#LeftBaseSVG").append("text")
                            .attr("class", "logBndOverlay")
                            .attr("x", mx).attr("y", my)
                            .attr("text-anchor", "middle")
                            .attr("dominant-baseline", "central")
                            .attr("font-size", "6")
                            .attr("fill", "#FF6D00")
                            .attr("stroke", "white")
                            .attr("stroke-width", "0.4")
                            .attr("paint-order", "stroke")
                            .text("W" + w.index);
                    });

                    // Dashed line from each room centre to nearest point on its closest wall
                    data.rooms.forEach(function(r) {
                        var cx = r.center_x, cy = r.center_y;
                        var w = data.wall_segments[r.closest_wall_idx];
                        var dx = w.x2 - w.x1, dy = w.y2 - w.y1;
                        var segLenSq = dx * dx + dy * dy;
                        var t = segLenSq === 0 ? 0 :
                            Math.max(0, Math.min(1, ((cx - w.x1) * dx + (cy - w.y1) * dy) / segLenSq));
                        var px = w.x1 + t * dx, py = w.y1 + t * dy;
                        d3.select("#LeftLayoutSVG").append("line")
                            .attr("class", "logBndOverlay")
                            .attr("x1", cx).attr("y1", cy)
                            .attr("x2", px).attr("y2", py)
                            .attr("stroke", "#FF6D00")
                            .attr("stroke-width", "1")
                            .attr("stroke-dasharray", "3,2");
                    });

                    var outsideCount = data.rooms.filter(function(r) { return !r.inside_boundary_extents; }).length;
                    var escapeCount  = data.rooms.filter(function(r) { return r.escape_area_px2 !== undefined && r.escape_area_px2 > 0.01; }).length;

                    alert("✅ Boundary Log Complete — see browser console for full detail\n\n" +
                        "Floor plan: " + data.floor_plan_id + "\n" +
                        "Boundary points: " + data.boundary_point_count + "\n" +
                        "Extents: x=[" + data.boundary_extents.x_min + ", " + data.boundary_extents.x_max + "]" +
                        "  y=[" + data.boundary_extents.y_min + ", " + data.boundary_extents.y_max + "]\n" +
                        "Rooms: " + data.room_count +
                        (outsideCount > 0 ? "\n⚠ " + outsideCount + " room(s) outside boundary extents" : "") +
                        (escapeCount  > 0 ? "\n⚠ " + escapeCount  + " room(s) partially outside boundary polygon" : ""));
                } else {
                    console.error("[Log Boundary] Failed:", data.error);
                    alert("❌ Log Boundary failed:\n" + data.error);
                }
            }).fail(function(xhr, status, error) {
                logBtn.innerHTML = originalText;
                logBtn.style.backgroundColor = "#00897B";
                logBtn.style.cursor = "pointer";
                console.error("[Log Boundary] Request failed:", xhr.responseText);
                alert("❌ Log Boundary request failed:\n" + error);
            });
        };

        // Export DXF button handler
        document.getElementById("exportDXFButton").onclick = function () {
            var dxfBtn = document.getElementById("exportDXFButton");
            var originalText = dxfBtn.innerHTML;
            dxfBtn.innerHTML = "⏳ Exporting...";
            dxfBtn.style.backgroundColor = "#757575";
            dxfBtn.style.cursor = "wait";

            $.get("/index/Export_DXF/", {}, function (data) {
                dxfBtn.innerHTML = originalText;
                dxfBtn.style.backgroundColor = "#2196F3";
                dxfBtn.style.cursor = "pointer";

                if (data.success) {
                    alert("✅ DXF Export Complete!\n\nFile: " + data.filename +
                          "\nSize: " + data.size_kb + " KB\nRooms: " + data.room_count);
                } else {
                    alert("❌ DXF Export failed:\n" + data.error);
                }
            }).fail(function (xhr, status, error) {
                dxfBtn.innerHTML = originalText;
                dxfBtn.style.backgroundColor = "#2196F3";
                dxfBtn.style.cursor = "pointer";
                try {
                    var resp = JSON.parse(xhr.responseText);
                    alert("❌ DXF Export failed:\n" + (resp.error || error));
                } catch (e) {
                    alert("❌ DXF Export request failed:\n" + error);
                }
            });
        };

        // Show Outside toggle handler
        document.getElementById("showOutsideButton").onclick = function () {
            var btn = this;
            var svg = d3.select("#LeftLayoutSVG");
            var revealed = btn.getAttribute("data-revealed") === "true";

            if (!revealed) {
                // REVEAL: strip clip-path, highlight rooms that extend outside boundary
                btn.setAttribute("data-revealed", "true");
                btn.innerHTML = "👁 Hide Outside";
                btn.style.backgroundColor = "#c62828";

                // Get boundary bounding box from the clipPath polygon in this SVG
                var clipPoly = document.querySelector("#LeftLayoutSVG clipPath polygon");
                var bndXMin = 0, bndXMax = 9999, bndYMin = 0, bndYMax = 9999;
                if (clipPoly) {
                    var pts = clipPoly.getAttribute("points").trim().split(/[\s,]+/);
                    var xs = [], ys = [];
                    for (var k = 0; k + 1 < pts.length; k += 2) {
                        xs.push(parseFloat(pts[k]));
                        ys.push(parseFloat(pts[k + 1]));
                    }
                    if (xs.length) {
                        bndXMin = Math.min.apply(null, xs);
                        bndXMax = Math.max.apply(null, xs);
                        bndYMin = Math.min.apply(null, ys);
                        bndYMax = Math.max.apply(null, ys);
                    }
                }

                // Strip clip-path from every room rect and mark rooms extending outside
                svg.selectAll("rect").each(function () {
                    var r = d3.select(this);
                    var cp = r.attr("clip-path");
                    if (cp) {
                        r.attr("data-orig-clip", cp).attr("clip-path", null);

                        var rx = parseFloat(r.attr("x"));
                        var ry = parseFloat(r.attr("y"));
                        var rw = parseFloat(r.attr("width"));
                        var rh = parseFloat(r.attr("height"));
                        var outside = rx < bndXMin - 0.5 || rx + rw > bndXMax + 0.5 ||
                                      ry < bndYMin - 0.5 || ry + rh > bndYMax + 0.5;
                        if (outside) {
                            svg.append("rect")
                                .attr("class", "outsideOverlay")
                                .attr("x", rx).attr("y", ry)
                                .attr("width", rw).attr("height", rh)
                                .attr("fill", "rgba(211,47,47,0.08)")
                                .attr("stroke", "#d32f2f")
                                .attr("stroke-width", 2)
                                .attr("stroke-dasharray", "6,3")
                                .attr("pointer-events", "none");
                        }
                    }
                });

            } else {
                // HIDE: restore clip-paths and remove overlays
                btn.setAttribute("data-revealed", "false");
                btn.innerHTML = "👁 Show Outside";
                btn.style.backgroundColor = "#6A1B9A";

                svg.selectAll("rect").each(function () {
                    var r = d3.select(this);
                    var orig = r.attr("data-orig-clip");
                    if (orig) {
                        r.attr("clip-path", orig).attr("data-orig-clip", null);
                    }
                });
                svg.selectAll(".outsideOverlay").remove();
            }
        };

        // Auto-Adjust button handler
        document.getElementById("AutoAdjust").onclick = function () {
            console.log("Auto-Adjust clicked!");

            // Get current graph state
            var currentGraph = GetEditGraph(0);

            // Get user requirements
            var obj = Num();
            var Numrooms = [];
            Numrooms.push(obj.roomactarr);
            Numrooms.push(obj.roomexaarr);
            Numrooms.push(obj.roomnumarr);

            console.log("Current graph:", currentGraph);
            console.log("Room requirements:", Numrooms);

            // Call backend to auto-adjust
            $.get("/index/AutoAdjustGraph/", {
                'NewGraph': JSON.stringify(currentGraph),
                'Numrooms': JSON.stringify(Numrooms)
            }, function (result) {
                console.log("Auto-adjust result:", result);

                // Clear existing graph
                d3.select('body').select('#LeftGraphSVG').selectAll('.TransLine').remove();
                d3.select('body').select('#LeftGraphSVG').selectAll('.TransCircle').remove();

                // Draw adjusted edges
                for (var i = 0; i < result.edges.length; i++) {
                    var u = result.edges[i][0];
                    var v = result.edges[i][1];

                    // Find node positions
                    var node_u = result.nodes.find(n => n[0] == u);
                    var node_v = result.nodes.find(n => n[0] == v);

                    if (node_u && node_v) {
                        var id = "TransLine_" + u + "_" + v + "_0";
                        CreateLine(node_u[2], node_u[3], node_v[2], node_v[3], id);
                    }
                }

                // Draw adjusted nodes
                for (var i = 0; i < result.nodes.length; i++) {
                    var indx = result.nodes[i][0];
                    var rmname = result.nodes[i][1];
                    var x = result.nodes[i][2];
                    var y = result.nodes[i][3];
                    var scalesize = result.nodes[i][4];

                    var id = "TransCircle_" + indx + "_" + rmname;
                    CreateCircle(x, y, id, 5);
                    d3.select("body").select("#LeftGraphSVG").select("#" + id).attr('scalesize', scalesize);
                }

                // Update room count cookie
                document.cookie = "RoomNum=" + result.nodes.length;

                console.log("Graph auto-adjusted successfully!");
                alert("Graph auto-adjusted! Added/removed rooms to match your requirements.");
            });
        };

        for (var i = 0; i < ret['hsedge'].length; i++) {
            var roomA = ret['hsedge'][i][0];
            var roomB = ret['hsedge'][i][1];
            var A_B = ret['hsedge'][i][2];
            var id = "TransLine" + "_" + roomA + "_" + roomB + "_" + A_B;

            CreateLine(ret['rmpos'][roomA][2], ret['rmpos'][roomA][3],
                ret['rmpos'][roomB][2], ret['rmpos'][roomB][3], id);
        }
        for (var i = 0; i < ret['rmpos'].length; i++) {

            var id = "TransCircle" + "_" + i + "_" + ret['rmpos'][i][1];
            CreateCircle(ret['rmpos'][i][2], ret['rmpos'][i][3], id, ret['rmsize'] [i][0][0]);
            d3.select("body").select("#LeftGraphSVG").select("#" + id).attr('scalesize', 1);

        }

        document.cookie = "RoomNum=" + ret['rmpos'].length;

        // COMMENTED OUT: Automatic floor plan generation after transfer
        // Now user must manually click "Show Plan" button to run AI model
        /*
        NewGraph = GetEditGraph(ret['rmpos']);
        $.get("/index/AdjustGraph/", {
            'NewGraph': JSON.stringify(NewGraph),
            'userRoomID': rooms.toString().split(',')[0],
            'adptRoomID': roomID
        }, function (adjust_ret) {
            // console.log("ret");
            CreateLeftPlan(adjust_ret['roomret'], adjust_ret['exterior'], adjust_ret["door"], adjust_ret["windows"], adjust_ret["indoor"], adjust_ret["windowsline"]);
        });
        */

        // downLoad button handler (moved outside of commented AJAX callback)
        document.getElementById("downLoad").onclick = function () {
                var arr, reg = new RegExp("(^| )hsname=([^;]*)(;|$)");
                if (arr = document.cookie.match(reg))
                    hsname = arr[2];
                console.log(focus_rect);
                if (document.getElementById("graph").checked == true)  {
                    var link = document.createElement('a');
                    link.href = "../static/" + hsname.split(".")[0] + ".mat";
                    var event = document.createEvent('MouseEvents');
                    event.initMouseEvent('click', true, false, window, 0, 0, 0, 0, 0, false, false, false, false, 0, null);
                    link.dispatchEvent(event);
                } else {
                    console.log("editing");
                    var NewLay = [];
                    NewLay = GetEditLayout();
                    var newGraph = [];
                    newGraph = GetEditGraph(ret['rmpos']);
                    $.get("/index/Save_Editbox/", {
                        'NewLay': JSON.stringify(NewLay),
                        'NewGraph': JSON.stringify(newGraph),
                        'userRoomID': rooms.toString().split(',')[0],
                        'adptRoomID': roomID
                    }, function (flag) {

                            var link = document.createElement('a');
                            link.href = "../static/" + hsname.split(".")[0] + ".png.mat";
                            var event = document.createEvent('MouseEvents');
                            event.initMouseEvent('click', true, false, window, 0, 0, 0, 0, 0, false, false, false, false, 0, null);
                            link.dispatchEvent(event);




                    });
                }

            }
        // }); // COMMENTED OUT: This was closing the automatic AdjustGraph AJAX call above

    });
    d3.select('body').select('#LeftGraphSVG').attr("transform", "scale(1.5)");

}

function showGraph(oCtl) {
    // $(oCtl).is(':checked') ? d3.select("body").select("#LeftGraphSVG").attr("opacity", "1.0") : d3.select("body").select("#LeftGraphSVG").attr("opacity", "0.0");
    // $(oCtl).is(':checked') ? d3.select("body").select("#LeftGraphSVG").attr("style", "position: relative;z-index:999!important;") : d3.select("body").select("#LeftGraphSVG").attr("style", "position: relative;z-index:888 !important;");
    // $(oCtl).is(':checked') ? d3.select("body").select("#LeftLayoutSVG").attr("style", "position: relative;margin-left: -259.5px;z-index:888!important;") : d3.select("body").select("#LeftLayoutSVG").attr("style", "position: relative;margin-left: -259.5px;z-index:999 !important;");
    if ($(oCtl).is(':checked')) {
        if (document.getElementById("layout").checked == true) {
            d3.select("body").select("#LeftGraphSVG").attr("display", "flex").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:999!important;");
            d3.select("body").select("#LeftLayoutSVG").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:888!important;").attr("opacity", "1.0");
        } else {
            d3.select("body").select("#LeftGraphSVG").attr("display", "flex").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:999!important;");
            d3.select("body").select("#LeftLayoutSVG").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:888!important;").attr("opacity", "0.0");
        }
        document.getElementById("graphimg").style = "display:inline-flex;";
        document.getElementById("graphdiv").style = "cursor: default;color: #000;width: 90px;border: 2px solid #0072ca;border-radius: 30px;text-align: center;vertical-align: middle;line-height: 26px;height: 30px;position: absolute;margin-left: 300px;"
        document.getElementById("Editing").style = "display:none;";

    } else {
        document.getElementById("graphimg").style = "display:none;";
        document.getElementById("Editing").style = "display:flex;margin-left: 140px;     margin-top: inherit;";

        // 方法一

        if (document.getElementById("layout").checked == true) {
            d3.select("body").select("#LeftGraphSVG").attr("display", "none").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:888!important;");
            d3.select("body").select("#LeftLayoutSVG").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:999!important;").attr("opacity", "1.0");
        } else {
            d3.select("body").select("#LeftGraphSVG").attr("display", "none").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:999!important;");
            d3.select("body").select("#LeftLayoutSVG").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:888!important;").attr("opacity", "0.0");
        }
        document.getElementById("graphdiv").style = "cursor: default;color: #000;width: 90px;        border: 2px solid #bfbfbf;border-radius: 30px;text-align: center;vertical-align: middle;line-height: 26px;height: 30px;position: absolute;margin-left: 300px;"

    }
}

function showRoom(oCtl) {
    // $(oCtl).is(':checked') ? d3.select("body").select("#LeftLayoutSVG").attr("opacity", "1.0") : d3.select("body").select("#LeftLayoutSVG").attr("opacity", "0.0");
    // $(oCtl).is(':checked') ? d3.select("body").select("#LeftLayoutSVG").attr("style", "position: relative;margin-left: -259.5px;z-index:888!important;") : d3.select("body").select("#LeftLayoutSVG").attr("style", "position: relative;margin-left: -259.5px;z-index:888!important;");
    // $(oCtl).is(':checked') ? d3.select("body").select("#LeftGraphSVG").attr("style", "position: relative;z-index:999!important;") : d3.select("body").select("#LeftGraphSVG").attr("style", "position: relative;z-index:999!important;");
    if ($(oCtl).is(':checked')) {
        if (document.getElementById("graph").checked == true) {
            d3.select("body").select("#LeftGraphSVG").attr("display", "flex").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:999!important;");
            d3.select("body").select("#LeftLayoutSVG").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:888!important;").attr("opacity", "1.0");
        } else {
            d3.select("body").select("#LeftGraphSVG").attr("display", "flex").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:888!important;");
            d3.select("body").select("#LeftLayoutSVG").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:999!important;").attr("opacity", "1.0");
        }
        document.getElementById("layoutimg").style = "display:inline-flex;";
        document.getElementById("layoutdiv").style = "cursor: default;color: #000;width: 90px;border: 2px solid #0072ca;border-radius: 30px;text-align: center;vertical-align: middle;line-height: 26px;height: 30px;position: absolute;margin-left: 400px;"

    } else {
        if (document.getElementById("graph").checked == true) {
            d3.select("body").select("#LeftGraphSVG").attr("display", "flex").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:999!important;");
            d3.select("body").select("#LeftLayoutSVG").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:888!important;").attr("opacity", "0.0");
        } else {
            d3.select("body").select("#LeftGraphSVG").attr("display", "none").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:999!important;");
            d3.select("body").select("#LeftLayoutSVG").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:888!important;").attr("opacity", "0.0");
        }
        document.getElementById("layoutimg").style = "display:none;";
        document.getElementById("layoutdiv").style = "cursor: default;color: #000;width: 90px;border: 2px solid #bfbfbf;border-radius: 30px;text-align: center;vertical-align: middle;line-height: 26px;height: 30px;position: absolute;margin-left: 400px;"

    }
}

function circle_mousedown() {
    console.log("circle_mousedown");

    if (createNewLine) {

        var id = "TransLine" + "_" + startPoint[0].split("_")[1] + "_" + this.id.split("_")[1] + "_0";

        if (hasLine(id)) {
            return;
        }

        //被选中的点现在也不显示是吧？这里变得颜色都一样
        var points = d3.select("body").select("#LeftGraphSVG").selectAll("circle").attr("stroke", "#000000").attr("stroke-width", 2);
        var selectPoint = d3.select("body").select("#LeftGraphSVG").select("#" + this.id).attr("stroke", "#000000").attr("stroke-width", 2);
        scalesize = d3.select("body").select("#LeftGraphSVG").select("#" + this.id).attr("scalesize");

        CreateLine(startPoint[1], startPoint[2], this.cx.animVal.value, this.cy.animVal.value, id);

        d3.select(this).remove();
        d3.select("#" + startPoint[0]).remove();
        adjust_graph = true;
        CreateCircle(startPoint[1], startPoint[2], startPoint[0], startPoint[3]);
        var start = d3.select("body").select("#LeftGraphSVG").select("#" + startPoint[0]).attr("scalesize", startPoint[4]);
        CreateCircle(this.cx.animVal.value, this.cy.animVal.value, this.id, selectPoint.attr('r'));
        var end = d3.select("body").select("#LeftGraphSVG").select("#" + this.id).attr("scalesize", scalesize);
        createNewLine = false;
        return;
    }

    focus_circle = true;
    dragging_circle = this; // Store reference to the circle being dragged
    var points = d3.select("body").select("#LeftGraphSVG").selectAll("circle").attr("stroke", "#000000").attr("stroke-width", 2);
    var selectPoint = d3.select("body").select("#LeftGraphSVG").select("#" + this.id).attr("stroke", "rgba(0,0,0,0.56)").attr("stroke-width", 2);
    var isDelete = document.querySelector('#isDelete');
    //禁用系统右键菜单
    document.oncontextmenu = function (eve) {
        return false;
    };

    if (d3.event.button == 2) {
        // var deletealert = confirm("是否删除？");
        // if (deletealert == true) {
        //     selectPoint.remove();
        //     focus_circle = false;
        //     adjust_graph = true;
        //     var pointInd = this.id.split("_")[1];
        //
        //     var lines = d3.select("body").select("#LeftGraphSVG").selectAll(".TransLine");
        //     lines.each(function (d, i) {
        //         var startPoint = this.id.split("_")[1];
        //         var endPoint = this.id.split("_")[2];
        //
        //         if (startPoint == pointInd || endPoint == pointInd) {
        //             adjust_graph = true;
        //             d3.select(this).remove();
        //         }
        //     })
        // }
        var leftsvg = document.getElementById('LeftGraphSVG');

//自定义右键菜单唤醒和关闭
        isDelete.style.left = (d3.event.clientX - 256) + 'px';
        isDelete.style.top = (d3.event.clientY) + 'px';
        isDelete.style.display = 'block';
        var pointInd = this.id.split("_")[1];

        //事件委托写法
        isDelete.onmousedown = function (eve) {

            if (eve.target.innerText == 'Delete') {
                setTimeout(function () {
                    selectPoint.remove();
                    focus_circle = false;
                    adjust_graph = true;

                    var lines = d3.select("body").select("#LeftGraphSVG").selectAll(".TransLine");
                    lines.each(function (d, i) {
                        var startPoint = this.id.split("_")[1];
                        var endPoint = this.id.split("_")[2];

                        if (startPoint == pointInd || endPoint == pointInd) {
                            adjust_graph = true;
                            d3.select(this).remove();
                        }
                    })
                }, 10);
            }
            if (eve.target.innerText == 'Scale*0.5') {
                SelectRadius = selectPoint.attr('r');
                ScaleRadius = SelectRadius * 0.5;
                selectPoint.attr('r', ScaleRadius);
                selectPoint.attr('scalesize', 0.5);
                console.log(selectPoint.attr('scalesize'));
            }
            if (eve.target.innerText == 'Scale*0.25') {
                SelectRadius = selectPoint.attr('r');
                ScaleRadius = SelectRadius * 0.25;
                selectPoint.attr('r', ScaleRadius);
                selectPoint.attr('scalesize', 0.25);
                console.log(selectPoint.attr('scalesize'));
            }
            if (eve.target.innerText == 'Scale*5') {
                SelectRadius = selectPoint.attr('r');
                ScaleRadius = SelectRadius * 5;
                selectPoint.attr('r', ScaleRadius);
                selectPoint.attr('scalesize', 5);
                console.log(selectPoint.attr('scalesize'));
            }
            if (eve.target.innerText == 'Scale*2') {
                SelectRadius = selectPoint.attr('r');
                ScaleRadius = SelectRadius * 2;
                selectPoint.attr('r', ScaleRadius);
                selectPoint.attr('scalesize', 2);
                console.log(selectPoint.attr('scalesize'));
            }

            isDelete.style.display = 'none';
        }
        $(document).click(function (e) {
            var pop = $('#isDelete')[0];
            if (e.target != pop && !$.contains(pop, e.target)) pop.style.display = 'none'
        })
    }

}

function hasLine(id) {
    var lines = d3.select(".TransLine");

    lines.each(function (d, i) {
        if (this.id == id)
            return true;
    });

    return false;
}

function circle_mousemove(event) {

    console.log("Move!");

    if (focus_circle && dragging_circle) {
        var leftsvg = document.getElementById('LeftGraphSVG');
        let newX = event.clientX - leftsvg.getBoundingClientRect().left;
        let newY = event.clientY - leftsvg.getBoundingClientRect().top;

        // console.log(newX + " " + newY)

        var transLines = d3.select("body").select("#LeftGraphSVG").selectAll(".TransLine");

        var pointID = (dragging_circle.id).split("_")[1];

        transLines.each(function (d, i) {
            var tmp_array = (this.id).split("_");

            if (tmp_array[1] == pointID) {
                d3.select(this).attr("x1", newX / 1.5).attr("y1", newY / 1.5);
            }
            if (tmp_array[2] == pointID) {
                d3.select(this).attr("x2", newX / 1.5).attr("y2", newY / 1.5);
            }
        })

        var selectPoint = d3.select("body").select("#LeftGraphSVG").select("#" + dragging_circle.id)
            .attr("cx", newX / 1.5).attr("cy", newY / 1.5);
        adjust_graph = true;
        // console.log(adjust_graph, "adjust")
    }
}

function circle_mouseup() {
    focus_circle = false;
    dragging_circle = null;
}

function circle_dblclick() {
    createNewLine = true;
    var selectPoint = d3.select("body").select("#LeftGraphSVG").select("#" + this.id).attr("stroke", "#d84447").attr("stroke-width", 3);

    startPoint[0] = this.id;
    startPoint[1] = this.cx.animVal.value;
    startPoint[2] = this.cy.animVal.value;
    console.log(this.r.animVal.value);
    startPoint[3] = this.r.animVal.value;
    startPoint[4] = this.attributes.scalesize.value;
}

function line_mousedown() {
    focus_line = true;
    var lines = d3.select("body").select("#LeftGraphSVG").selectAll("line").attr("stroke", "#000000")
    var selectLine = d3.select("body").select("#LeftGraphSVG").select("#" + this.id).attr("stroke", "#d83230");

    if (d3.event.button == 2) {

        //var startPoint = this.id.split("_")[1];
        //var endPoint = this.id.split("_")[2];

        //console.log(startPoint,endPoint);

        //var isStartSingle = true;
        //var isEndSingle = true;

        selectLine.remove();

        //var curlines = d3.select("body").select("#LeftGraphSVG").selectAll("line");

        /*curlines.each(function(d,i){
            var tmp_start = this.id.split("_")[1];
            var tmp_end = this.id.split("_")[2];

            if(startPoint == tmp_start || startPoint == tmp_end) isStartSingle = false;
            if(endPoint == tmp_start || endPoint == tmp_end) isEndSingle = false;
        })

        console.log(isStartSingle,isEndSingle);

        var circles = d3.select("body").select("#LeftGraphSVG").selectAll(".TransCircle");

        circles.each(function(d,i){
            var tmp_ind = this.id.split("_")[1];

            if(isStartSingle && tmp_ind==startPoint) d3.select(this).remove();
            if(isEndSingle && tmp_ind==endPoint) d3.select(this).remove();
        })*/

        focus_line = false;
    }
}

function line_mouseup() {
    focus_line = false;
}

function rect_mousedown() {
    console.log("rect_mousedown");

    if (focus_rect != "") {
        var leftlaysvg = document.getElementById('LeftLayoutSVG');

        let mousex = (d3.event.x - leftlaysvg.getBoundingClientRect().left) / 2;
        let mousey = (d3.event.y - leftlaysvg.getBoundingClientRect().top) / 2;
        var oldx = startRectvalue[0];
        var oldy = startRectvalue[1];
        var oldw = startRectvalue[2];
        var oldh = startRectvalue[3];

        Type = rectzoomType(mousex, mousey, oldx, oldy, oldw, oldh);
        console.log(Type);

        rect_type = true;
    }
}

function rectzoomType(mousex, mousey, oldx, oldy, oldw, oldh) {
    if (oldy < mousey && mousey < oldy + oldh) {
        if (mousex < oldx + oldw + 16) {
            // $('#LeftLayoutSVG').css('cursor', 'e-resize');
            if (oldx + oldw - 16 < mousex) {
                d3.select("body").select("#LeftLayoutSVG").attr('cursor', 'e-resize');
                var type = "right";
                return type;
            }
        }
        if (mousex < oldx + 12) {
            if (oldx - 16 < mousex) {
                d3.select("body").select("#LeftLayoutSVG").attr('cursor', 'w-resize');
                var type = "left";
                return type;

            }

        }
    }
    if (oldx < mousex && oldx < oldx + oldw) {
        if (mousey < oldy + 16) {
            if (oldy - 16 < mousey) {
                d3.select("body").select("#LeftLayoutSVG").attr('cursor', 'n-resize');
                var type = "top";
                return type;
            }


        }
        if (mousey < oldy + oldh + 16) {
            if (oldy + oldh - 16 < mousey) {
                d3.select("body").select("#LeftLayoutSVG").attr('cursor', 's-resize');
                var type = "down";
                return type;

            }

        }
    }
    if (type == undefined) {
        d3.select("body").select("#LeftLayoutSVG").attr('cursor', 'default');

    }
}

function rect_mousemove() {

    var leftlaysvg = document.getElementById('LeftLayoutSVG');
    let mousex = (d3.event.x - leftlaysvg.getBoundingClientRect().left) / 2;
    let mousey = (d3.event.y - leftlaysvg.getBoundingClientRect().top) / 2;
        console.log("rect_mousemove",mousex,mousey);
    console.log(focus_rect);
    if (focus_rect == "dblclick") {
        // var oldx = this.x.animVal.value;
        // var oldy = this.y.animVal.value;
        //
        // var oldw = this.width.animVal.value;
        // var oldh = this.height.animVal.value;
        var oldx = startRectvalue[0];
        var oldy = startRectvalue[1];
        var oldw = startRectvalue[2];
        var oldh = startRectvalue[3];
        rectzoomType(mousex, mousey, oldx, oldy, oldw, oldh);
        // console.log(type);
        if (rect_type) {
            var item = null;
            var obj = document.getElementsByName("edit");
            for (var i = 0; i < obj.length; i++) { //遍历Radio
                if (obj[i].checked) {
                    item = obj[i].value;
                }
            }

            if (item == "local") {
                switch (Type) {
                    case "right":
                        selectRect.attr("width", mousex - oldx);
                        break;
                    case "left":
                        // selectRect.attr("x", mousex).attr("width", mousex - oldx + oldw);
                        selectRect.attr("x", mousex);
                        selectRect.attr("width", oldx - mousex + oldw);
                        break;
                    case "top":
                        selectRect.attr("y", mousey).attr("height", oldy - mousey + oldh);
                        break;
                    case "down":
                        selectRect.attr("height", mousey - oldy);
                        break;

                }

            }
            if (item == "global") {
                switch (Type) {
                    case "right":
                        for (i = 0; i < RelRectvalue[0].length; i++) {
                            var RelRect = d3.select("body").select("#LeftLayoutSVG").select("#" + RelRectvalue[0][i][4]);
                            RelRect.attr("width", mousex - RelRectvalue[0][i][0]);
                        }
                        break;
                    case "left":
                        for (i = 0; i < RelRectvalue[1].length; i++) {
                            var RelRect = d3.select("body").select("#LeftLayoutSVG").select("#" + RelRectvalue[1][i][4]);
                            RelRect.attr("x", mousex);
                            RelRect.attr("width", Number(RelRectvalue[1][i][0]) - mousex + Number(RelRectvalue[1][i][2]));
                        }
                        break;

                    case "down":
                        for (i = 0; i < RelRectvalue[3].length; i++) {
                            var RelRect = d3.select("body").select("#LeftLayoutSVG").select("#" + RelRectvalue[3][i][4]);
                            RelRect.attr("height", mousey - RelRectvalue[3][i][1]);
                        }
                        break;

                    case "top":

                        for (i = 0; i < RelRectvalue[2].length; i++) {
                            console.log(RelRectvalue[2][i]);
                            var RelRect = d3.select("body").select("#LeftLayoutSVG").select("#" + RelRectvalue[2][i][4]);
                            RelRect.attr("y", mousey).attr("height", Number(RelRectvalue[2][i][1]) - mousey + Number(RelRectvalue[2][i][3]));
                        }
                        break;
                }

            }

        }
    }
}


function rect_mouseup() {
    console.log("rect_mouseup");
    focus_rect = "";
    rect_type = false;
    var interior_color = roomcolor("Interior wall");

    var rects = d3.select("body").select("#LeftLayoutSVG").selectAll("rect").attr("stroke", interior_color).attr("stroke-width", 4);
    d3.select("body").select("#LeftLayoutSVG").attr('cursor', 'default');

}

function rect_dblclick() {
    console.log("rect_dblclick");
    var item = null;
    var obj = document.getElementsByName("edit");
    for (var i = 0; i < obj.length; i++) { //遍历Radio
        if (obj[i].checked) {
            item = obj[i].value;
        }
    }
    var interior_color = roomcolor("Interior wall");
    var rects = d3.select("body").select("#LeftLayoutSVG").selectAll("rect").attr("stroke", interior_color).attr("stroke-width", 4);
    selectRect = d3.select("body").select("#LeftLayoutSVG").select("#" + this.id).attr("stroke", "#d84447").attr("stroke-width", 4);
    focus_rect = "dblclick";


    startRectvalue[0] = this.x.animVal.value;
    startRectvalue[1] = this.y.animVal.value;
    startRectvalue[2] = this.width.animVal.value;
    startRectvalue[3] = this.height.animVal.value;
    if (item == "global") {
        console.log(this.id);
        $.get("/index/RelBox/", {
            'selectRect': this.id

        }, function (rdirgroup) {

            for (k = 0; k < rdirgroup.length; k++) {
                var RelRectvalue2 = [];
                for (i = 0; i < rdirgroup[k].length; i++) {
                    var RelRectvalue1 = [];

                    var RelRect = d3.select("body").select("#LeftLayoutSVG").select("#" + rdirgroup[k][i]);
                    RelRectvalue1[0] = RelRect.attr("x");
                    RelRectvalue1[1] = RelRect.attr("y");
                    RelRectvalue1[2] = RelRect.attr("width");
                    RelRectvalue1[3] = RelRect.attr("height");
                    RelRectvalue1[4] = rdirgroup[k][i];
                    RelRectvalue2[i] = RelRectvalue1
                }
                RelRectvalue[k] = RelRectvalue2;
            }
            console.log(RelRectvalue);
            console.log(RelRectvalue[2]);

            console.log(RelRectvalue[0]);

        });
    }
}

function rect_click() {
    console.log("rect_click");
    focus_rect = "click";

}
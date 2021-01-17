$("document").ready(function () {
  var total = 11;
  function myTable(data) {
    var tb = document.getElementById("myTable");
    for (var k = 1; k <= 5; k++) {
      tb.rows[k].cells[0].innerHTML = data[k - 1].id;
      tb.rows[k].cells[1].innerHTML = data[k - 1].firstName;
      tb.rows[k].cells[2].innerHTML = data[k - 1].lastName;
      tb.rows[k].cells[3].innerHTML = data[k - 1].username;
      tb.rows[k].cells[4].innerHTML = data[k - 1].email;
    }
  }
  function goToPage(pageNum) {
    $.ajax({
      url: "http://localhost:5000/user/page=" + pageNum,
      type: "GET",
      //contentType: "applicaton/json",
      //dataType: "json",
    }).done(function (data) {
      document.getElementById("per_page").innerHTML = data.users_per_page;
      total = data.total_users;
      document.getElementById("total").innerHTML = total;
      myTable(data.data);
    });
  }

  var idDelete;
  var idEdit;

  $(".delete").click(function () {
    idDelete = $(this).parent().siblings()[0].innerHTML;
    //idds.css({ color: "red", border: "2px solid red" });
  });
  $(".delete-user").click(function () {
    deleteUser(idDelete);
  });
  $(".search-user").click(function () {
    var id = parseInt($("#search-id").val());

    searchUser(id);
  });
  $("#back").click(function () {
    $("#searchTable").hide();
    $("#myTable").show();
    $("#paging").show();
  });
  function searchUser(id) {
    $.ajax({
      url: "http://localhost:5000/user/" + id,
      type: "GET",
      success: function (response) {
        console.log("deleted");
        console.log(response);
        $("#searchTable").show();
        $("#myTable").hide();
        $("#paging").hide();
        var tb = document.getElementById("mySearchTable");
        tb.rows[1].cells[0].innerHTML = response.id;
        tb.rows[1].cells[1].innerHTML = response.firstName;
        tb.rows[1].cells[2].innerHTML = response.lastName;
        tb.rows[1].cells[3].innerHTML = response.username;
        tb.rows[1].cells[4].innerHTML = response.email;
      },
    });
  }
  function deleteUser(id) {
    $.ajax({
      url: "http://localhost:5000/user/" + id,
      type: "DELETE",
      success: function (response) {
        console.log("deleted");
        console.log(response);
        location.reload(true);
      },
    });
  }
  $(".edit").click(function () {
    idEdit = $(this).parent().siblings()[0].innerHTML;
    //idds.css({ color: "red", border: "2px solid red" });
  });
  $(".edit-user").click(function () {
    editUser(idEdit);
  });
  function editUser(id) {
    var message = {};
    message["firstName"] = $("#edit-first-name").val();
    message["lastName"] = $("#edit-last-name").val();
    message["username"] = $("#edit-user-name").val();
    message["email"] = $("#edit-email").val();

    $.ajax({
      url: "http://localhost:5000/user/",
      type: "POST",
      contentType: "applicaton/json",
      data: JSON.stringify(message),
      success: function (response) {
        console.log("updated");
        console.log(response);
        location.reload(true);
      },
    });
  }

  $(".add-user").click(function () {
    addUser();
  });

  function addUser() {
    var message = {};
    message["firstName"] = $("#add-first-name").val();
    message["lastName"] = $("#add-last-name").val();
    message["username"] = $("#add-user-name").val();
    message["email"] = $("#add-email").val();

    $.ajax({
      url: "http://localhost:5000/user",
      type: "POST",
      contentType: "applicaton/json",
      data: JSON.stringify(message),
      success: function (response) {
        console.log("updated");
        console.log(response);
        location.reload(true);
      },
    });
  }

  (function (name) {
    var container = $(name);
    $("#searchTable").hide();
    var options = {
      callback: function (response, pagination) {
        window.console && console.log(response, pagination);
        goToPage(pagination.pageNumber);
      },
      dataSource: function (done) {
        $.ajax({
          type: "GET",
          url: "http://localhost:5000/user",
          success: function (response) {
            var foo = [];

            for (var i = 1; i <= response.total_users; i++) {
              foo.push(i);
            }

            done(foo);
          },
        });
      },
      className: "paginationjs-theme-blue paginationjs-medium",
      pageSize: 5,
    };

    //$.pagination(container, options);

    container.addHook("beforeInit", function () {
      window.console && console.log("beforeInit...");
    });
    container.pagination(options);

    container.addHook("beforePageOnClick", function () {
      window.console && console.log("beforePageOnClick...");
      //return false
    });
  })("#page");
});

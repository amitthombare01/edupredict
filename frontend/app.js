const API = "http://127.0.0.1:8000";

// ---------------- LOGIN ----------------

document.getElementById("login").onsubmit = async (e)=>{
e.preventDefault();

document.getElementById("login-form").classList.add("hidden");
document.getElementById("dashboard").classList.remove("hidden");

loadData();
};

// ---------------- CSV UPLOAD ----------------

document.getElementById("upload-form").onsubmit = async (e)=>{
e.preventDefault();

const file = document.getElementById("csv-file").files[0];

const formData = new FormData();
formData.append("file", file);

const res = await fetch(API+"/api/upload-csv",{
method:"POST",
body:formData
});

const data = await res.json();

alert("Uploaded! Accuracy: "+data.accuracy);

loadData();
};

// ---------------- LOAD DATA ----------------

async function loadData(){

const res = await fetch(API+"/api/students");
const data = await res.json();

renderTable(data.records, data.columns);
renderInsights(data.records);
renderChart(data.records);
}

// ---------------- TABLE ----------------

function renderTable(data, columns){

const table = document.getElementById("students");
const tbody = table.querySelector("tbody");
tbody.innerHTML="";

if(!data.length) return;

table.classList.remove("hidden");

// dynamic columns
const thead = table.querySelector("thead tr");
thead.innerHTML="";

columns.forEach(col=>{
thead.innerHTML += `<th>${col}</th>`;
});

data.forEach(row=>{
let tr="<tr>";
columns.forEach(col=>{
tr += `<td>${row[col] ?? ""}</td>`;
});
tr+="</tr>";
tbody.innerHTML+=tr;
});
}

// ---------------- INSIGHTS ----------------

function renderInsights(data){

if(!data.length) return;

const keys = Object.keys(data[0]);

const nameCol = keys.find(k=>k.toLowerCase().includes("name")) || keys[0];
const predCol = keys.find(k=>k.toLowerCase().includes("prediction"));

document.getElementById("top-students").innerHTML =
data.slice(0,3).map(s=>s[nameCol]).join("<br>");

if(predCol){
document.getElementById("risk-students").innerHTML =
data.filter(s=>s[predCol]==0)
.map(s=>s[nameCol]).join("<br>") || "None";
}else{
document.getElementById("risk-students").innerText="No prediction";
}

}

// ---------------- CHART ----------------

let chart;

function renderChart(data){

const ctx = document.getElementById("scoreChart");

if(chart) chart.destroy();

const numericKey = Object.keys(data[0]).find(k=>typeof data[0][k]==="number");

chart = new Chart(ctx,{
type:"bar",
data:{
labels:data.map((_,i)=>"S"+(i+1)),
datasets:[{
label:numericKey,
data:data.map(d=>d[numericKey])
}]
}
});
}
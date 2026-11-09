// Potree viewer for the Aulas 3 digital twin.
//
// The page comes from the API (same origin as /spaces and /chat); the point
// clouds come from the scan server, whose address /spaces reports. Each floor
// is its own cloud, so floors can be shown or hidden one at a time.
//
// The "Medir salones" panel turns Potree measurements into the marker and
// dimensions of each space and downloads an updated spaces.json, so nobody has
// to type coordinates by hand.

const LABELS = {
	co2: ["CO₂", "ppm"],
	temperature: ["Temperatura", "°C"],
	humidity: ["Humedad", "%"],
	light: ["Luz", "%"],
	motion: ["Movimiento", ""],
};
const SURVEY_KEY = "aulas3-survey";
// The survey panel (markers and room sizes) is a tool for whoever maps the
// building, not for the people asking questions. It only appears when the
// page is opened with ?medir in the address.
const SURVEY_MODE = new URLSearchParams(location.search).has("medir");
// The API names question types in English (code identifiers); the person
// reading the chat sees them in Spanish.
const INTENT_LABELS = {
	current: "lectura actual",
	forecast: "pronóstico",
	history: "histórico",
	anomaly: "anomalía",
};

const state = {
	spaces: [],
	clouds: {},          // floor -> PointCloudOctree
	annotations: {},     // space_id -> Annotation
	selected: null,
	survey: loadSurvey(), // space_id -> {marker, dimensions}, kept until downloaded
};

const $id = (id) => document.getElementById(id);

window.viewer = new Potree.Viewer($id("potree_render_area"));
viewer.setEDLEnabled(true);
viewer.setFOV(60);
// The RTX 4090 handles far more than Potree's default of one million.
viewer.setPointBudget(5_000_000);
viewer.setBackground("gradient");
viewer.loadGUI(() => {
	viewer.setLanguage("es");
	$("#menu_tools").next().show();
});

main();

async function main() {
	let registry;
	try {
		registry = await (await fetch("/spaces")).json();
	} catch (e) {
		setStatus(`No pude leer /spaces: ${e}`);
		return;
	}
	state.spaces = registry.spaces;
	if (SURVEY_MODE) {
		fillSpaces();
		setupSurvey();
	} else {
		$id("survey").remove();
	}
	await loadFloors(registry.scans_url, registry.floors);
	drawMarkers();
	if (SURVEY_MODE) updateSurveyCount();
}

// --------------------------------------------------------------------------
// Floors
// --------------------------------------------------------------------------
async function loadFloors(scansUrl, floors) {
	const container = $id("floors");
	let loaded = 0;
	for (const floor of floors) {
		const url = `${scansUrl}/${floor.cloud}`;
		const label = document.createElement("label");
		container.append(label);

		// Potree's loader never settles on a missing cloud, so check first.
		const ok = await fetch(url, { method: "HEAD" }).then(r => r.ok).catch(() => false);
		if (!ok) {
			label.className = "missing";
			label.textContent = `Piso ${floor.floor}: sin convertir`;
			continue;
		}

		const box = document.createElement("input");
		box.type = "checkbox";
		box.checked = true;
		box.dataset.floor = floor.floor;
		label.append(box, ` Piso ${floor.floor}`);

		const e = await Potree.loadPointCloud(url, `Piso ${floor.floor}`);
		const cloud = e.pointcloud;
		cloud.material.activeAttributeName = "rgba";
		cloud.material.pointSizeType = Potree.PointSizeType.ADAPTIVE;
		cloud.material.size = 1;
		viewer.scene.addPointCloud(cloud);
		state.clouds[floor.floor] = cloud;
		box.addEventListener("change", () => { cloud.visible = box.checked; });
		loaded++;
	}
	// Frame the whole building once every floor is in, not just the first one.
	if (loaded) viewer.fitToScreen();
	setStatus(loaded
		? `${loaded} de ${floors.length} pisos cargados desde ${scansUrl}`
		: `Ningún piso disponible en ${scansUrl}. ¿Está corriendo python -m scans.server?`);
}

// A question about one floor leaves only that floor visible, so the answer
// and the scene talk about the same thing. The checkboxes follow, so the user
// can turn the others back on.
function showOnlyFloor(floorNumber) {
	for (const [floor, cloud] of Object.entries(state.clouds)) {
		cloud.visible = Number(floor) === floorNumber;
	}
	for (const box of document.querySelectorAll("#floors input[type=checkbox]")) {
		box.checked = box.dataset.floor === String(floorNumber);
	}
}

function showAllFloors() {
	for (const cloud of Object.values(state.clouds)) cloud.visible = true;
	for (const box of document.querySelectorAll("#floors input[type=checkbox]")) box.checked = true;
}

// --------------------------------------------------------------------------
// Spaces and markers
// --------------------------------------------------------------------------
function fillSpaces() {
	const select = $id("survey-space");
	select.innerHTML = "<option value=''>Elige el salón que mediste</option>";
	for (const space of state.spaces) {
		const option = document.createElement("option");
		option.value = space.space_id;
		const sensor = Object.keys(space.readings).length ? " • con sensor" : "";
		option.textContent = `${space.room ?? space.name} (piso ${space.floor})${sensor}`;
		select.append(option);
	}
	select.addEventListener("change", () => select.value && selectSpace(Number(select.value)));
}

// Unsaved survey edits show only in survey mode; everyone else sees the
// markers already written to spaces.json.
function markerOf(space) {
	return (SURVEY_MODE ? state.survey[space.space_id]?.marker : null) ?? space.marker;
}

function drawMarkers() {
	for (const a of Object.values(state.annotations)) viewer.scene.annotations.remove(a);
	state.annotations = {};
	for (const space of state.spaces) {
		const marker = markerOf(space);
		if (!marker) continue;
		const a = new Potree.Annotation({
			position: marker,
			title: space.room ?? space.name,
			description: describe(space),
		});
		a.addEventListener("click", () => (SURVEY_MODE ? selectSpace : highlightMarker)(space.space_id));
		viewer.scene.annotations.add(a);
		state.annotations[space.space_id] = a;
	}
}

function describe(space) {
	return Object.entries(space.readings)
		.filter(([, v]) => v !== null)
		.map(([k, v]) => `${LABELS[k]?.[0] ?? k}: ${v} ${LABELS[k]?.[1] ?? ""}`)
		.join("<br>");
}

// The selected space only drives the survey panel. The chat never uses it:
// a question has to name its room or floor, so the answer is never about a
// space the user did not ask for.
function selectSpace(spaceId) {
	state.selected = state.spaces.find(s => s.space_id === spaceId) ?? null;
	$id("survey-space").value = spaceId;
	highlightMarker(spaceId);
	showSurveyFor(state.selected);
}

function highlightMarker(spaceId) {
	for (const [id, a] of Object.entries(state.annotations)) {
		a.domElement.toggleClass("selected", Number(id) === spaceId);
	}
}

// --------------------------------------------------------------------------
// Chat
// --------------------------------------------------------------------------
$id("ask").addEventListener("submit", async (event) => {
	event.preventDefault();
	const text = $id("question").value.trim();
	if (!text) return;
	$id("question").value = "";
	addMessage("user", text);
	const pending = addMessage("bot", "Pensando…");

	try {
		const r = await fetch("/chat", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ question: text }),
		});
		const answer = await r.json();
		if (!r.ok) throw new Error(JSON.stringify(answer.detail ?? answer));
		pending.textContent = answer.response;
		const room = state.spaces.find(s => s.space_id === answer.space);
		const isFloor = answer.floor !== null && answer.floor !== undefined;
		const about = answer.building ? "todo el edificio"
			: isFloor ? (answer.floor === 0 ? "planta baja" : `piso ${answer.floor}`)
			: (room?.room ?? room?.name ?? answer.space);
		// No meta line when the API answered without routing (help, which
		// room, unknown room).
		if (about !== null && about !== undefined) {
			const kind = INTENT_LABELS[answer.intent_type] ?? answer.intent_type;
			appendSpan(pending, "meta", `${kind} • ${about}`
				+ (answer.forecast_method ? ` • ${answer.forecast_method}, ${answer.horizon_min} min` : ""));
		}
		if (answer.warning) appendSpan(pending, "warning", answer.warning);
		if (answer.building) showAllFloors();
		else if (isFloor) showOnlyFloor(answer.floor);
		else if (room) highlightMarker(room.space_id);
	} catch (e) {
		pending.textContent = `Error al consultar la API: ${e.message}`;
	}
});

$id("question").addEventListener("keydown", (event) => {
	if (event.key === "Enter" && !event.shiftKey) {
		event.preventDefault();
		$id("ask").requestSubmit();
	}
});

function addMessage(kind, text) {
	const div = document.createElement("div");
	div.className = `msg ${kind}`;
	div.textContent = text;
	$id("messages").append(div);
	div.scrollIntoView({ block: "end" });
	return div;
}

function appendSpan(parent, cls, text) {
	const span = document.createElement("span");
	span.className = cls;
	span.textContent = text;
	parent.append(span);
}

// --------------------------------------------------------------------------
// Survey: markers and dimensions measured in Potree
// --------------------------------------------------------------------------
function lastMeasure(minPoints) {
	const measures = viewer.scene.measurements.filter(m => m.points.length >= minPoints);
	return measures[measures.length - 1] ?? null;
}

function round(v) {
	return Math.round(v * 1000) / 1000;
}

// Length of a measurement: vertical span for Potree's height tool, path length
// otherwise.
function measureLength(m) {
	const p = m.points.map(x => x.position);
	if (m.showHeight) {
		const z = p.map(v => v.z);
		return Math.max(...z) - Math.min(...z);
	}
	let total = 0;
	for (let i = 1; i < p.length; i++) total += p[i].distanceTo(p[i - 1]);
	return total;
}

function setupSurvey() {
	$id("take-marker").addEventListener("click", () => {
		const m = lastMeasure(1);
		if (!m) return alert("Primero mide un punto con la herramienta de Potree.");
		const v = m.points[m.points.length - 1].position;
		$id("marker-value").textContent = [v.x, v.y, v.z].map(round).join(", ");
		$id("marker-value").dataset.value = JSON.stringify([v.x, v.y, v.z].map(round));
	});

	for (const button of document.querySelectorAll("#survey [data-dim]")) {
		button.addEventListener("click", () => {
			const m = lastMeasure(2);
			if (!m) return alert("Primero mide una distancia con la herramienta de Potree.");
			$id(`dim-${button.dataset.dim}`).value = round(measureLength(m));
		});
	}

	$id("save-space").addEventListener("click", () => {
		const space = state.selected;
		if (!space) return alert("Elige primero, en la lista de este panel, el salón que mediste.");
		const marker = $id("marker-value").dataset.value;
		const dims = ["length", "width", "height"].map(k => parseFloat($id(`dim-${k}`).value));
		state.survey[space.space_id] = {
			marker: marker ? JSON.parse(marker) : markerOf(space),
			dimensions: dims.every(Number.isFinite)
				? { length: dims[0], width: dims[1], height: dims[2] }
				: (state.survey[space.space_id]?.dimensions ?? space.dimensions),
		};
		saveSurvey();
		drawMarkers();
		selectSpace(space.space_id);
	});

	function showSurveyFor(space) {
		const saved = space ? (state.survey[space.space_id] ?? space) : {};
		const marker = saved.marker;
		$id("marker-value").textContent = marker ? marker.join(", ") : "—";
		$id("marker-value").dataset.value = marker ? JSON.stringify(marker) : "";
		for (const k of ["length", "width", "height"]) {
			$id(`dim-${k}`).value = saved.dimensions?.[k] ?? "";
		}
	}

	$id("download").addEventListener("click", async () => {
		const registry = await (await fetch("/spaces/registry")).json();
		for (const space of registry.spaces) {
			const edit = state.survey[space.space_id];
			if (edit) Object.assign(space, edit);
		}
		const blob = new Blob([JSON.stringify(registry, null, 2) + "\n"], { type: "application/json" });
		const link = document.createElement("a");
		link.href = URL.createObjectURL(blob);
		link.download = "spaces.json";
		link.click();
		URL.revokeObjectURL(link.href);
	});
}

function updateSurveyCount() {
	const n = Object.keys(state.survey).length;
	$id("survey-count").textContent = n
		? `${n} espacio(s) medidos sin descargar. Reemplaza config/spaces.json con el archivo descargado.`
		: "";
}

// Measurements survive a page reload until they are written to spaces.json.
function loadSurvey() {
	try { return JSON.parse(localStorage.getItem(SURVEY_KEY)) ?? {}; } catch { return {}; }
}

function saveSurvey() {
	try { localStorage.setItem(SURVEY_KEY, JSON.stringify(state.survey)); } catch { /* private window */ }
	updateSurveyCount();
}

function setStatus(text) {
	$id("status").textContent = text;
}

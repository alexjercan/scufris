// The `/api/stats` and `/api/host/overview` wire shapes: the metrics the stats
// dashboard renders, and the host inspection reports beside them.

export interface MemStats {
    total: number;
    used: number;
    available: number;
    percent: number;
}

export interface SwapStats {
    total: number;
    used: number;
    percent: number;
}

export interface DiskUsage {
    mountpoint: string;
    total: number;
    used: number;
    percent: number;
}

export interface NetIO {
    bytes_sent: number;
    bytes_recv: number;
}

export interface GpuStats {
    name: string;
    util_percent: number;
    mem_used_mb: number;
    mem_total_mb: number;
    mem_percent: number;
    temp_c: number;
    power_w: number;
    power_limit_w: number;
    clock_sm_mhz: number;
    clock_mem_mhz: number;
}

export interface SensorReading {
    label: string;
    current: number;
    high: number | null;
    critical: number | null;
}

export interface SensorGroup {
    chip: string;
    readings: SensorReading[];
}

export interface FanReading {
    chip: string;
    label: string;
    rpm: number;
}

export interface NetIfRate {
    name: string;
    sent_per_sec: number;
    recv_per_sec: number;
}

export interface DiskIoRate {
    name: string;
    read_per_sec: number;
    write_per_sec: number;
}

export interface CpuActivity {
    ctx_switches_per_sec: number;
    interrupts_per_sec: number;
}

// Mirrors scufris.metrics.HostStats (the /api/stats payload).
export interface HostStats {
    hostname: string;
    os_name: string;
    kernel: string;
    cpu_percent: number;
    per_cpu_percent: number[];
    mem: MemStats;
    swap: SwapStats;
    disks: DiskUsage[];
    load_avg: [number, number, number];
    uptime_seconds: number;
    net: NetIO;
    sampled_at: string;
    gpus: GpuStats[];
    temps: SensorGroup[];
    fans: FanReading[];
    per_cpu_freq_mhz: number[];
    net_interfaces: NetIfRate[];
    disk_io: DiskIoRate[];
    process_count: number;
    cpu_activity: CpuActivity;
}

// --- host inspection (/api/host/overview) -----------------------------------
//
// Mirrors scufris/host: every report carries its own availability, so the UI
// renders a REASON when something could not be read instead of an empty card
// that reads as "nothing wrong". See that package's docstring.

export interface Availability {
    ok: boolean;
    reason: string;
    caveat: string;
}

export interface UnitSummary {
    name: string;
    load: string;
    active: string;
    sub: string;
    description: string;
}

export interface UnitList {
    available: Availability;
    scope: string;
    state_filter: string;
    units: UnitSummary[];
    truncated: boolean;
}

export interface Generation {
    number: number;
    date: string;
    nixos_version: string;
    kernel_version: string;
    configuration_revision: string;
    current: boolean;
}

export interface FilesystemUsage {
    mountpoint: string;
    device: string;
    fstype: string;
    total: number;
    used: number;
    free: number;
    percent: number;
}

export interface StorageReport {
    available: Availability;
    filesystems: { available: Availability; filesystems: FilesystemUsage[] };
    generations: { available: Availability; generations: Generation[] };
    nix_store: FilesystemUsage | null;
}

export interface ThrottleCounters {
    available: Availability;
    // Per PHYSICAL core (hyperthread siblings share one counter), not per
    // logical cpu - hence both cores_read and cpus_read.
    core_events: number;
    package_events: number;
    core_time_ms: number;
    package_time_ms: number;
    cpus_read: number;
    cores_read: number;
    cores_throttled: number;
}

export interface HostTemperature {
    chip: string;
    label: string;
    celsius: number;
    high: number | null;
    critical: number | null;
}

export interface ThermalReport {
    available: Availability;
    temperatures: HostTemperature[];
    throttling: ThrottleCounters;
    battery: {
        available: Availability;
        present: boolean;
        percent: number | null;
    };
    fans: { available: Availability; present: boolean };
}

export interface HostOverview {
    failed_system_units: UnitList;
    failed_user_units: UnitList;
    storage: StorageReport;
    thermal: ThermalReport;
}

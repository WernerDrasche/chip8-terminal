const std = @import("std");
const Allocator = std.mem.Allocator;
const Random = std.rand.Random;
const ArrayList = std.ArrayList;
const HashMap = std.AutoHashMap;
const Thread = std.Thread;
const Mutex = std.Thread.Mutex.AtomicMutex;
const mem = std.mem;
const testing = std.testing;
const log = std.log;
const linux = std.os.linux;
const json = std.json;

//TODO: implement sound (no clue how to do that yet)

pub const log_level = std.log.Level.err;

const VTIME = 5;
const VMIN = 6;

const hexafont = [16][5]u8{
    .{0xf0, 0x90, 0x90, 0x90, 0xf0}, // 0
    .{0x20, 0x60, 0x20, 0x20, 0x70}, // 1
    .{0xf0, 0x10, 0xf0, 0x80, 0xf0}, // 2
    .{0xf0, 0x10, 0xf0, 0x10, 0xf0}, // 3
    .{0x90, 0x90, 0xf0, 0x10, 0x10}, // 4
    .{0xf0, 0x80, 0xf0, 0x10, 0xf0}, // 5
    .{0xf0, 0x80, 0xf0, 0x90, 0xf0}, // 6
    .{0xf0, 0x10, 0x20, 0x40, 0x40}, // 7
    .{0xf0, 0x90, 0xf0, 0x90, 0xf0}, // 8
    .{0xf0, 0x90, 0xf0, 0x10, 0xf0}, // 9
    .{0xf0, 0x90, 0xf0, 0x90, 0x90}, // A
    .{0xe0, 0x90, 0xe0, 0x90, 0xe0}, // B
    .{0xf0, 0x80, 0x80, 0x80, 0xf0}, // C
    .{0xe0, 0x90, 0x90, 0x90, 0xe0}, // D
    .{0xf0, 0x80, 0xf0, 0x80, 0xf0}, // E
    .{0xf0, 0x80, 0xf0, 0x80, 0x80}, // F
};

const Coord = struct {
    x: u8,
    y: u8,
};

const Settings = struct {
    sleep_secs: u64 = 0,
    default_sleep_secs: u64 = 0,
    sleep_nanos: u64 = 2088333,
    default_sleep_nanos: u64 = 2088333,
    map: HashMap(u8, u8),

    error_flag: bool = false,

    fn init(args: Args, allocator: Allocator) !Settings {
        var map = HashMap(u8, u8).init(allocator);
        try map.put('1', 0x1);
        try map.put('2', 0x2);
        try map.put('3', 0x3);
        try map.put('4', 0x4);
        try map.put('5', 0x5);
        try map.put('6', 0x6);
        try map.put('7', 0x7);
        try map.put('8', 0x8);
        try map.put('9', 0x9);
        try map.put('a', 0xa);
        try map.put('b', 0xb);
        try map.put('c', 0xc);
        try map.put('d', 0xd);
        try map.put('e', 0xe);
        try map.put('f', 0xf);
        var result = Settings{
            .map = map,
        };
        const cwd = std.fs.cwd();
        // max_size is arbitrary
        const file = cwd.readFileAlloc(allocator, "config.json", 4096) catch {
            result.error_flag = true;
            log.warn("couldn't open config.json", .{});
            return result;
        };
        defer allocator.free(file);
        var parser = json.Parser.init(allocator, false);
        defer parser.deinit();
        var tree = try parser.parse(file);
        defer tree.deinit();
        const prog_names = [_][]const u8{"default", args.name};
        for (prog_names) |name| {
            if (tree.root.Object.get(name)) |default| {
                if (default.Object.get("sleep_secs")) |val| {
                    const sleep_secs: u64 = if (val.Integer < 0) blk: {
                        result.error_flag = true;
                        log.err("invalid time for sleep_secs", .{});
                        break :blk result.default_sleep_secs;
                    } else @intCast(u64, val.Integer);
                    result.default_sleep_secs = sleep_secs;
                    result.sleep_secs = sleep_secs;
                }
                if (default.Object.get("sleep_nanos")) |val| {
                    const sleep_nanos = if (val.Integer < 0) blk: {
                        result.error_flag = true;
                        log.err("invalid time for sleep_nanos", .{});
                        break :blk result.default_sleep_nanos;
                    } else @intCast(u64, val.Integer);
                    result.default_sleep_nanos = sleep_nanos;
                    result.sleep_nanos = sleep_nanos;
                }
                if (default.Object.get("keymap")) |keymap| {
                    var iter = keymap.Object.iterator();
                    while (iter.next()) |entry| {
                        if (entry.key_ptr.len != 1) {
                            log.err("invalid length for mapped key", .{});
                            result.error_flag = true;
                            continue;
                        }
                        const key = entry.key_ptr.ptr[0];
                        const to_str = entry.value_ptr.String;
                        const to = std.fmt.parseInt(u8, to_str, 16) catch {
                            result.error_flag = true;
                            log.err("could not parse {d}", .{to_str});
                            continue;
                        };
                        try map.put(key, to);
                    }
                }
            }
        }
        return result;
    }

    fn deinit(self: *Settings) void {
        self.map.deinit();
    }
};

var settings: Settings = undefined;

fn nextNotEmpty(comptime T: type, iterator: *mem.SplitIterator(T)) ?[]const T {
    while (iterator.next()) |slice| {
        if (slice.len > 0) {
            log.debug("got {s}", .{slice});
            return slice;
        } 
    } else return null;
}

fn printMap(writer: anytype, map: anytype) !void {
    var iter = map.iterator();
    while (iter.next()) |entry| {
        try writer.print("{c} -> {x}\n", .{entry.key_ptr.*, entry.value_ptr.*});
    }
}

const VideoRAM = struct {
    const WIDTH = 64;
    const HEIGHT = 32;

    const BYTE_WIDTH = @divExact(WIDTH, 8);

    buf: [HEIGHT][BYTE_WIDTH]u8,
    updated: bool = true,

    fn init() VideoRAM {
        return .{
            .buf = [_][BYTE_WIDTH]u8{ [_]u8{0} ** BYTE_WIDTH } ** HEIGHT,
        };
    }

    fn clear(self: *VideoRAM, writer: anytype) !void {
        self.updated = false;
        try moveCursorTo(writer, 0, 0);
        for (self.buf) |*row, y| {
            for (row) |*byte, x| {
                var i: usize = 0;
                while (i < 8) : (i += 1) {
                    if (@shlWithOverflow(u8, byte.*, 1, byte)) {
                        try moveCursorTo(writer, (x * 8) + i, y);
                        try writer.writeAll("\x1b[40m  ");
                    } 
                }
            }
        }
        try moveCursorHome(writer);
    }

    fn wrappedXOR(self: *VideoRAM, writer: anytype, pos: Coord, byte: u8) !bool {
        var x = pos.x % WIDTH;
        var y = pos.y % HEIGHT;
        const splitoff = x % 8;
        var i: u8 = 0;
        var l: u8 = byte;
        var r: u8 = 0;
        while (i < splitoff) : (i += 1) {
            r >>= 1;
            if (l & 0x01 != 0) {
                r |= 0x80;
            }
            l >>= 1;
        }
        const xl = x / 8;
        const xr = (xl + 1) % BYTE_WIDTH;
        const wrap = xr < xl;
        const l_ptr = &self.buf[y][xl]; 
        const r_ptr = &self.buf[y][xr];
        const collision = ((l_ptr.* & l) | (r_ptr.* & r)) != 0;
        l_ptr.* ^= l;
        r_ptr.* ^= r;
        l = l_ptr.* << @intCast(u3, splitoff);
        r = r_ptr.*;
        try moveCursorTo(writer, x, y);
        try drawPixels(writer, l, 8 - splitoff);
        if (wrap) {
            try moveCursorTo(writer, 0, y);
        }
        try drawPixels(writer, r, splitoff);
        return collision;
    }

    fn drawSprite(self: *VideoRAM, writer: anytype, pos: Coord, sprite: []const u8) !bool {
        const x = pos.x;
        const y = pos.y;
        var collision = false;
        for (sprite) |byte, i| {
            collision = (try self.wrappedXOR(writer, .{.x=x, .y=y+@intCast(u8, i)}, byte)) or collision;
        }
        try writer.writeAll("\x1b[0m");
        try moveCursorHome(writer);
        return collision;
    }

    fn drawScreen(self: *VideoRAM, writer: anytype) !void {
        try writer.writeAll("\x1b[0m\x1b[H\x1b[J");
        try writer.writeAll("-" ** ((WIDTH * 2) + 2) ++ "\n");
        for (self.buf) |row| {
            try writer.writeByte('|');
            for (row) |byte| {
                try drawPixels(writer, byte, 8);
            }
            try writer.writeAll("\x1b[0m|\n");
        }
        try writer.writeAll("-" ** ((WIDTH * 2) + 2));
        try moveCursorHome(writer);
    }

    fn drawPixels(writer: anytype, byte: u8, amount: u8) !void {
        var b = byte;
        var i: usize = 0;
        while (i < amount) : (i += 1) {
            if (@shlWithOverflow(u8, b, 1, &b)) {
                try writer.writeAll("\x1b[47m  ");
            } else {
                try writer.writeAll("\x1b[40m  ");
            }
        }
        //std.os.nanosleep(0, 50000000);
    }

    fn moveCursorHome(writer: anytype) !void {
        try moveCursorTo(writer, 0, HEIGHT + 2);
        try writer.writeAll("\x1b[1G\x1b[0J");
    }

    fn moveCursorTo(writer: anytype, x: usize, y: usize) !void {
        try writer.print("\x1b[{d};{d}f", .{y + 2, (x * 2) + 2});
    }
};

const Keypad = struct {
    const CYCLES_RETAIN = 240;

    queue: [CYCLES_RETAIN]u8 = [_]u8{0} ** CYCLES_RETAIN,
    idx: usize = 0,

    fn readByteToInputQueue(self: *Keypad, reader: anytype) !void {
        self.idx = (self.idx + 1) % CYCLES_RETAIN;
        self.queue[self.idx] = reader.readByte() catch 0;
    }

    fn readByteFromInputQueue(self: *Keypad) ?u8 {
        var i: usize = 0;
        while (i < CYCLES_RETAIN) : (i += 1) {
            const byte = self.queue[self.idx];
            if (byte != 0) {
                return if (byte == 255) null else byte;
            }
            if (self.idx == 0) {
                self.idx = CYCLES_RETAIN;
            }
            self.idx -= 1;
        }
        return null;
    }

    fn consumeKeypress(self: *Keypad) void {
        self.queue[self.idx] = 255;
    }

    fn isPressed(self: *Keypad, key: u8) bool {
        const byte = self.readByteFromInputQueue() orelse return false;
        const input = settings.map.get(byte) orelse return false;
        //log.debug("pressed=0x{x} expected=0x{x}", .{input, key});
        const match = input == key;
        if (match) {
            self.consumeKeypress();
        }
        return match;
    }

    fn getPressed(reader: anytype) !?u8 {
        const byte = reader.readByte() catch return null;
        if (byte == 0x1b) {
            return error.Escape;
        }
        return settings.map.get(byte);
    }
};

const Chip8CPU = struct {
    const MEM_SIZE = 4096;

    v: [16]u8 = .{0} ** 16,
    ptr: u16 = 0,
    mem: [MEM_SIZE]u8 = .{0} ** MEM_SIZE,

    ip: u16 = 0,
    stack: ArrayList(u16),

    dt: u8 = 255,
    dt_mutex: Mutex, 
    dt_thread: Thread = undefined,
    st: u8 = 0, // does nothing
    st_mutex: Mutex,
    //st_thread: Thread = undefined,
    counter_term_flag: bool = false,

    video: *VideoRAM,
    input: *Keypad,
    rng: Random,
    program_size: u16 = 0,

    fn init(allocator: Allocator, rng: Random, video: *VideoRAM, input: *Keypad) Chip8CPU {
        var cpu = Chip8CPU{
            .stack = ArrayList(u16).init(allocator),
            .video = video,
            .input = input,
            .rng = rng,
            .dt_mutex = Mutex{},
            .st_mutex = Mutex{},
        };
        for (hexafont) |*sprite, i| {
            mem.copy(u8, cpu.mem[(i * 5)..], sprite);
        }
        return cpu;
    }

    fn deinit(self: *Chip8CPU) void {
        self.stack.deinit();
    }

    fn startCounters(self: *Chip8CPU) !void {
        self.dt_thread = 
            try Thread.spawn(.{}, countDown60Hz, .{&self.dt, &self.dt_mutex, &self.counter_term_flag});
    }

    fn stopCounters(self: *Chip8CPU) void {
        self.counter_term_flag = true;
        self.dt_thread.join();
    }

    fn countDown60Hz(reg: *u8, mutex: *Mutex, term_flag: *bool) void {
        while (true) {
            if (term_flag.*) break;
            mutex.lock();
            if (reg.* > 0) {
                reg.* -= 1;
            }
            mutex.unlock();
            std.os.nanosleep(0, 16666666);
        }
    }

    fn loadProgram(self: *Chip8CPU, start_address: u16, program: []u8) void {
        mem.copy(u8, self.mem[start_address..], program);
        self.ip = start_address;
        self.program_size = @intCast(u16, program.len);
        self.stack.clearRetainingCapacity();
    }

    fn stepInstruction(
        self: *Chip8CPU,
        reader: anytype,
        writer: anytype,
        canonical_term: *linux.termios,
        allocator: Allocator,
    ) !State {
        const instruction = blk: {
            var result: u16 = self.mem[self.ip];
            result <<= 8;
            break :blk result | self.mem[self.ip + 1];
        };
        self.ip += 2;
        const prefix: u8 = @intCast(u8, instruction >> 12);
        const suffix3: u16 = instruction & 0x0fff;
        const suffix2: u8 = @intCast(u8, instruction & 0x00ff);
        const suffix1: u8 = @intCast(u8, instruction & 0x000f);
        const x = (instruction & 0x0f00) >> 8;
        const y = (instruction & 0x00f0) >> 4;
        const vx = &self.v[x];
        const vy = &self.v[y];
        const vf = &self.v[0xf];
        switch (prefix) {
            0x0 => {
                switch (suffix2) {
                    0xe0 => try self.video.clear(writer),
                    0xee => self.ip = self.stack.pop(),
                    0xfd => return .halt,
                    else => return error.UnknownInstruction,
                }
            },
            0x1 => self.ip = suffix3,
            0x2 => {
                try self.stack.append(self.ip);
                self.ip = suffix3;
            },
            0x3 => if (vx.* == suffix2) { self.ip += 2; },
            0x4 => if (vx.* != suffix2) { self.ip += 2; },
            0x5 => if (vx.* == vy.*) { self.ip += 2; },
            0x6 => vx.* = suffix2,
            0x7 => _ = @addWithOverflow(u8, vx.*, suffix2, vx),
            0x8 => {
                switch (suffix1) {
                    0x0 => vx.* = vy.*,
                    0x1 => vx.* |= vy.*,
                    0x2 => vx.* &= vy.*,
                    0x3 => vx.* ^= vy.*,
                    0x4 => vf.* = if (@addWithOverflow(u8, vx.*, vy.*, vx)) 1 else 0,
                    0x5 => vf.* = if (@subWithOverflow(u8, vx.*, vy.*, vx)) 0 else 1,
                    0x6 => {
                        vf.* = vx.* & 0x01;
                        vx.* >>= 1;
                    },
                    0x7 => vf.* = if (@subWithOverflow(u8, vy.*, vx.*, vx)) 0 else 1,
                    0xe => vf.* = if (@shlWithOverflow(u8, vx.*, 1, vx)) 1 else 0,
                    else => return error.UnknownInstruction,
                }
            },
            0x9 => if (suffix1 == 0x0 and vx.* != vy.*) { self.ip += 2; },
            0xa => self.ptr = suffix3,
            0xb => self.ip = suffix3 + self.v[0],
            0xc => vx.* = self.rng.int(u8) & suffix2,
            0xd => {
                const sprite = self.mem[self.ptr..(self.ptr + suffix1)];
                vf.* = if (try self.video.drawSprite(writer, .{.x=vx.*, .y=vy.*}, sprite)) 1 else 0;
            },
            0xe => {
                switch (suffix2) {
                    0x9e => if (self.input.isPressed(vx.*)) { self.ip += 2; },
                    0xa1 => if (!self.input.isPressed(vx.*)) { self.ip += 2; },
                    else => return error.UnknownInstruction,
                }
            },
            0xf => {
                switch (suffix2) {
                    0x07 => vx.* = self.dt,
                    0x0a => {
                        while (true) {
                            const press = Keypad.getPressed(reader) catch blk: {
                                const state = try commandMode(reader, writer, canonical_term, self.video, allocator);
                                if (state != .ok) {
                                    return state;
                                } else {
                                    break :blk null;
                                }
                            };
                            if (press) |input| {
                                vx.* = input;
                                break;
                            }
                        }
                    },
                    0x15 => {
                        self.dt_mutex.lock();
                        self.dt = vx.*;
                        self.dt_mutex.unlock();
                    },
                    0x18 => {
                        self.st_mutex.lock();
                        self.st = vx.*;
                        self.st_mutex.unlock();
                    },
                    0x1e => self.ptr += vx.*,
                    0x29 => self.ptr = @intCast(u16, vx.* & 0x0f) * 5,
                    0x33 => {
                        var bcd = [3]u8{0, 0, 0};
                        var n = vx.*;
                        var i: usize = 0;
                        while (n > 0) : (n /= 10) {
                            i += 1;
                            bcd[bcd.len - i] = n % 10;
                        }
                        mem.copy(u8, self.mem[self.ptr..], &bcd);
                    },
                    0x55 => mem.copy(u8, self.mem[self.ptr..], self.v[0..(x + 1)]),
                    0x65 => {
                        mem.copy(u8, &self.v, self.mem[self.ptr..(self.ptr + x + 1)]);
                    },
                    else => return error.UnknownInstruction,
                }
            },
            else => return error.UnknownInstruction,
        }
        _ = suffix1;
        return .ok;
    }
};

const Args = struct {
    start_address: u16,
    program: []u8,
    name: []const u8,
    allocator: Allocator,

    fn collect(allocator: Allocator) !Args {
        const args = try std.process.argsAlloc(allocator);
        defer std.process.argsFree(allocator, args);
        var start_address: u16 = 0x200;
        var filename: []const u8 = "default.ch8";
        const cwd = std.fs.cwd();
        for (args[1..]) |arg| {
            if (mem.eql(u8, arg, "--e") or mem.eql(u8, arg, "--eti-660")) {
                start_address = 0x600;
            } else {
                filename = arg;
            }
        }
        const dot_ind = if (mem.indexOf(u8, filename, ".")) |ind| ind else filename.len;
        const basename = filename[0..dot_ind];
        var name = try allocator.alloc(u8, basename.len);
        mem.copy(u8, name, basename);
        const max_size = Chip8CPU.MEM_SIZE - start_address;
        const program = try cwd.readFileAlloc(allocator, filename, max_size);
        return Args{
            .start_address = start_address,
            .program = program,
            .name = name,
            .allocator = allocator,
        };
    }

    fn deinit(self: Args) void {
        self.allocator.free(self.program);
    }
};

const State = enum {
    ok,
    halt,
    restart,
};

fn commandMode(
    reader: anytype,
    writer: anytype,
    canonical_term: *linux.termios,
    video: *VideoRAM,
    allocator: Allocator,
) !State {
    var buf: [100]u8 = undefined;
    const parse_err_msg: []const u8 = "not a valid integer";
    const invalid_err_msg: []const u8 = "invalid command";
    var err_msg = invalid_err_msg;
    exitRawMode(canonical_term, allocator);
    try writer.writeAll("\x1b[0K");
    defer _ = enterRawMode(allocator);
    while (true) : (if (err_msg.len > 0) log.err("{s}", .{err_msg})) {
        err_msg = invalid_err_msg;
        try writer.writeAll("command: ");
        var input = try reader.readUntilDelimiter(&buf, '\n');
        var iter = mem.split(u8, input, " ");
        log.debug("{s}", .{iter});
        var command = nextNotEmpty(u8, &iter) orelse continue;
        log.debug("entering {s} command", .{command});
        if (mem.lastIndexOfScalar(u8, command, 0x1b)) |i| {
            command = command[i+1..];
        }
        if (mem.eql(u8, command, "nanosleep")) {
            const secs_str = nextNotEmpty(u8, &iter) orelse continue;
            const secs = std.fmt.parseInt(u64, secs_str, 10) catch {
                err_msg = parse_err_msg;
                continue;
            };
            const nanos_str = nextNotEmpty(u8, &iter) orelse continue;
            const nanos = std.fmt.parseInt(u64, nanos_str, 10) catch {
                err_msg = parse_err_msg;
                continue;
            };
            settings.sleep_secs = secs;
            settings.sleep_nanos = nanos;
            err_msg = "";
        } else if (mem.eql(u8, command, "speed")) {
            const level = nextNotEmpty(u8, &iter) orelse continue;
            if (mem.eql(u8, level, "default")) {
                settings.sleep_nanos = settings.default_sleep_nanos;
                settings.sleep_secs = settings.default_sleep_secs;
            } else if (mem.eql(u8, level, "faster")) {
                settings.sleep_nanos /= 2;
            } else if (mem.eql(u8, level, "slower")) {
                settings.sleep_nanos *= 2;
            }
            err_msg = "";
        } else if (mem.eql(u8, command, "remap")) {
            const slice = nextNotEmpty(u8, &iter) orelse continue;
            if (slice.len > 1) continue;
            const key = slice[0];
            const to_str = nextNotEmpty(u8, &iter) orelse continue;
            const to = std.fmt.parseInt(u8, to_str, 16) catch {
                err_msg = parse_err_msg;
                continue;
            };
            try settings.map.put(key, to);
            err_msg = "";
        } else if (mem.eql(u8, command, "view")) {
            const obj = nextNotEmpty(u8, &iter) orelse continue;
            if (mem.eql(u8, obj, "keymap")) {
                try printMap(writer, settings.map);
            } else if (mem.eql(u8, obj, "sleep_nanos")) {
                try writer.print("{d}\n", .{settings.sleep_nanos});
            } else if (mem.eql(u8, obj, "sleep_secs")) {
                try writer.print("{d}\n", .{settings.sleep_secs});
                
            } else {
                err_msg = "setting doesn't exist";
                continue;
            }
            err_msg = "";
        } else if (mem.eql(u8, command, "exit")) {
            return .halt;
        } else if (mem.eql(u8, command, "restart")) {
            log.debug("restarting", .{});
            return .restart;
        } else if (mem.eql(u8, command, "resume")) {
            log.debug("resuming", .{});
            break;
        } else {
            continue;
        }
    }
    try video.drawScreen(writer);
    return .ok;
}

fn enterRawMode(allocator: Allocator) linux.termios {
    _ = std.ChildProcess.exec(.{.allocator=allocator, .argv=&[_][]const u8{"xset", "r", "rate", "100"}}) catch {};
    var tty_attr: linux.termios = undefined;
    _ = linux.tcgetattr(linux.STDIN_FILENO, &tty_attr);
    const tty_attr_bak = tty_attr;
    tty_attr.lflag &= (~(linux.ICANON|linux.ECHO));
    tty_attr.cc[VTIME] = 0;
    tty_attr.cc[VMIN] = 0;
    _ = linux.tcsetattr(linux.STDIN_FILENO, linux.TCSA.NOW, &tty_attr);
    return tty_attr_bak;
}

fn exitRawMode(tty_attr_bak: *linux.termios, allocator: Allocator) void {
    _ = std.ChildProcess.exec(.{.allocator=allocator, .argv=&[_][]const u8{"xset", "r", "rate"}}) catch {};
    std.os.nanosleep(0, 100000000);
    _ = linux.tcsetattr(linux.STDIN_FILENO, linux.TCSA.NOW, tty_attr_bak);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var rng = std.rand.DefaultPrng.init(@intCast(u64, std.time.milliTimestamp())).random();
    const allocator = gpa.allocator();
    const stdout = std.io.getStdOut().writer();
    const stdin = std.io.getStdIn().reader();
    const args = try Args.collect(allocator);
    defer args.deinit();
    settings = try Settings.init(args, allocator);
    defer settings.deinit();
    if (settings.error_flag) {
        try stdout.print("press [ENTER] to continue", .{});
        try stdin.skipUntilDelimiterOrEof('\n');
    }
    var vram = VideoRAM.init();
    try vram.drawScreen(stdout);
    var keypad = Keypad{};
    var cpu = Chip8CPU.init(allocator, rng, &vram, &keypad);
    defer cpu.deinit();
    try cpu.startCounters();
    defer cpu.stopCounters();
    cpu.loadProgram(args.start_address, args.program);
    var tty_attr_bak = enterRawMode(allocator);
    defer exitRawMode(&tty_attr_bak, allocator);
    var status: State = .ok;
    while (status == .ok) {
        try keypad.readByteToInputQueue(stdin);
        if (keypad.readByteFromInputQueue()) |byte| {
            if (byte == 0x1b) {
                keypad.consumeKeypress();
                status = try commandMode(stdin, stdout, &tty_attr_bak, &vram, allocator);
            }
        }
        switch (status) {
            .ok => {
                status = cpu.stepInstruction(stdin, stdout, &tty_attr_bak, allocator) catch blk: {
                    try stdout.print("{}", .{cpu});
                    break :blk .halt;
                };
                std.os.nanosleep(settings.sleep_secs, settings.sleep_nanos);
            },
            .restart => {
                cpu.loadProgram(args.start_address, args.program);
                try vram.clear(stdout);
                try vram.drawScreen(stdout);
                status = .ok;
            },
            else => {},
        }
    }
}

import { memo, useCallback, useEffect, useId, useState } from "react";
import {
  Bookmark,
  ChevronUp,
  MessageSquarePlus,
  MoreHorizontal,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  Pencil,
  Pin,
  PinOff,
  Scale,
  Search,
  Settings,
  Sun,
  Trash2,
  Users,
  X,
} from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { usePersistentState } from "@/lib/legal/storage";
import type { Conversation } from "@/lib/legal/types";
import { BrandLockup } from "./BrandLogo";

export type WorkspaceView = "qa" | "search" | "saved";

const NAV: { id: WorkspaceView; label: string; icon: typeof Scale }[] = [
  { id: "qa", label: "Hỏi đáp pháp luật", icon: Scale },
  { id: "search", label: "Tra cứu văn bản", icon: Search },
  { id: "saved", label: "Văn bản đã lưu", icon: Bookmark },
];

function groupConversations(list: Conversation[]) {
  const DAY = 86_400_000;
  const now = Date.now();
  const groups: { label: string; items: Conversation[] }[] = [
    { label: "Đã ghim", items: [] },
    { label: "Hôm nay", items: [] },
    { label: "7 ngày qua", items: [] },
    { label: "Cũ hơn", items: [] },
  ];
  for (const c of list) {
    if (c.pinned) groups[0].items.push(c);
    else if (now - c.updatedAt < DAY) groups[1].items.push(c);
    else if (now - c.updatedAt < DAY * 7) groups[2].items.push(c);
    else groups[3].items.push(c);
  }
  return groups.filter((g) => g.items.length > 0);
}

interface SidebarProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  /** Mobile drawer uses the same content but closes instead of collapsing. */
  inDrawer?: boolean;
  /** Prevents a flash animation while persisted desktop state is hydrating. */
  motionReady?: boolean;
  view: WorkspaceView;
  onViewChange: (v: WorkspaceView) => void;
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onRename: (id: string) => void;
  onPin: (id: string) => void;
  onDelete: (id: string) => void;
  theme: "light" | "dark";
  onToggleTheme: () => void;
  onOpenSettings: () => void;
}

export function ThemeToggle({
  theme,
  onToggle,
  collapsed,
}: {
  theme: "light" | "dark";
  onToggle: () => void;
  collapsed?: boolean;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={onToggle}
          aria-label={theme === "dark" ? "Chuyển sang giao diện sáng" : "Chuyển sang giao diện tối"}
          className={cn(
            "sidebar-utility-button inline-flex h-9 items-center gap-2.5 rounded-lg px-2.5 text-[12.5px] font-medium text-sidebar-foreground transition-colors hover:bg-sidebar-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none",
            collapsed ? "size-10 justify-center px-0" : "w-full",
          )}
        >
          {theme === "dark" ? (
            <Sun className="size-4 shrink-0" aria-hidden />
          ) : (
            <Moon className="size-4 shrink-0" aria-hidden />
          )}
          {!collapsed && <span>{theme === "dark" ? "Giao diện sáng" : "Giao diện tối"}</span>}
        </button>
      </TooltipTrigger>
      {collapsed && <TooltipContent side="right">Đổi giao diện</TooltipContent>}
    </Tooltip>
  );
}

export const ConversationHistory = memo(function ConversationHistory({
  conversations,
  activeId,
  collapsed,
  onSelect,
  onRename,
  onPin,
  onDelete,
}: Pick<
  SidebarProps,
  "conversations" | "activeId" | "collapsed" | "onSelect" | "onRename" | "onPin" | "onDelete"
>) {
  const groups = groupConversations(conversations);

  return (
    <nav
      aria-label="Lịch sử trò chuyện"
      aria-hidden={collapsed}
      className={cn(
        "sidebar-history-panel w-[271px] space-y-4 px-2 pb-2",
        collapsed && "sidebar-expanded-hidden",
      )}
    >
      {groups.map((g) => (
        <div key={g.label}>
          <p className="px-2 pb-1.5 text-[10px] font-semibold tracking-[0.09em] text-muted-foreground uppercase">
            {g.label}
          </p>
          <ul className="space-y-0.5">
            {g.items.map((c) => {
              const active = c.id === activeId;
              return (
                <li
                  key={c.id}
                  data-active={active ? "true" : "false"}
                  className="sidebar-history-row group relative"
                >
                  <button
                    type="button"
                    onClick={() => onSelect(c.id)}
                    aria-current={active ? "page" : undefined}
                    className={cn(
                      "sidebar-history-item flex h-8 w-full items-center gap-2 rounded-lg pr-8 pl-2.5 text-left text-[12.5px] transition-colors focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none",
                      active
                        ? "sidebar-history-item-active font-medium text-sidebar-accent-foreground"
                        : "text-sidebar-foreground/85 hover:bg-sidebar-accent/60",
                    )}
                  >
                    <span
                      className={cn(
                        "sidebar-history-indicator h-3.5 w-0.5 shrink-0 rounded-full",
                        active ? "bg-[var(--brand)]" : "bg-transparent",
                      )}
                    />
                    <span className="truncate">{c.title}</span>
                    {c.pinned && (
                      <Pin className="ml-auto size-3 shrink-0 text-[var(--brand)]" aria-hidden />
                    )}
                  </button>

                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <button
                        type="button"
                        aria-label={`Tùy chọn cho ${c.title}`}
                        className="sidebar-history-menu absolute top-1/2 right-1 inline-flex size-6 -translate-y-1/2 items-center justify-center rounded-md text-muted-foreground transition-[opacity,color,background-color] hover:bg-background/70 hover:text-foreground focus-visible:opacity-100 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
                      >
                        <MoreHorizontal className="size-3.5" aria-hidden />
                      </button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end" className="w-44">
                      <DropdownMenuItem onSelect={() => onRename(c.id)}>
                        <Pencil className="size-3.5" aria-hidden /> Đổi tên
                      </DropdownMenuItem>
                      <DropdownMenuItem onSelect={() => onPin(c.id)}>
                        {c.pinned ? (
                          <>
                            <PinOff className="size-3.5" aria-hidden /> Bỏ ghim
                          </>
                        ) : (
                          <>
                            <Pin className="size-3.5" aria-hidden /> Ghim
                          </>
                        )}
                      </DropdownMenuItem>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem
                        onSelect={() => onDelete(c.id)}
                        className="text-[var(--flag)]"
                      >
                        <Trash2 className="size-3.5" aria-hidden /> Xóa
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </li>
              );
            })}
          </ul>
        </div>
      ))}
    </nav>
  );
});

ConversationHistory.displayName = "ConversationHistory";

export function Sidebar(props: SidebarProps) {
  const {
    collapsed,
    onToggleCollapse,
    inDrawer = false,
    motionReady = true,
    view,
    onViewChange,
    conversations,
    activeId,
    onSelect,
    onNew,
    onRename,
    onPin,
    onDelete,
    theme,
    onToggleTheme,
    onOpenSettings,
  } = props;
  const sidebarId = useId();

  return (
    <aside
      id={sidebarId}
      data-collapsed={collapsed ? "true" : "false"}
      data-motion-ready={motionReady ? "true" : "false"}
      aria-label="Thanh điều hướng HCMUTE-SHIPCODE"
      className={cn(
        "sidebar-shell flex h-full flex-col border-r border-sidebar-border bg-sidebar transition-[width] duration-150 ease-[cubic-bezier(0.2,0,0,1)] motion-reduce:transition-none",
        inDrawer
          ? "w-full overflow-x-hidden overflow-y-auto"
          : collapsed
            ? "w-[68px] overflow-hidden"
            : "w-[272px] overflow-hidden",
      )}
    >
      <header
        className={cn(
          "sidebar-brand-header flex h-14 shrink-0 items-center px-3.5",
          collapsed && "justify-center px-0",
        )}
      >
        {collapsed ? (
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                onClick={onToggleCollapse}
                aria-label="Mở rộng thanh bên"
                aria-expanded={!collapsed}
                aria-controls={sidebarId}
                className="sidebar-brand-reopen group relative inline-flex size-11 items-center justify-center rounded-xl focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
              >
                <BrandLockup collapsed size={32} />
                <span className="sidebar-reopen-cue" aria-hidden>
                  <PanelLeftOpen className="size-2.5" />
                </span>
              </button>
            </TooltipTrigger>
            <TooltipContent side="right">Mở rộng thanh bên</TooltipContent>
          </Tooltip>
        ) : (
          <>
            <BrandLockup size={32} />
            <button
              type="button"
              onClick={onToggleCollapse}
              aria-label={inDrawer ? "Đóng menu điều hướng" : "Thu gọn thanh bên"}
              aria-expanded={inDrawer ? undefined : !collapsed}
              aria-controls={sidebarId}
              className="ml-2 inline-flex size-7 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-[color,background-color,transform] duration-150 hover:bg-sidebar-accent hover:text-foreground active:scale-95 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
            >
              {inDrawer ? (
                <X className="size-3.5" aria-hidden />
              ) : (
                <PanelLeftClose className="size-3.5" aria-hidden />
              )}
            </button>
          </>
        )}
      </header>

      <div className="px-3 pt-3 pb-2">
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              type="button"
              onClick={onNew}
              aria-label="Cuộc trò chuyện mới"
              className={cn(
                "sidebar-new-chat inline-flex items-center justify-center gap-2.5 rounded-[10px] bg-primary text-[12.5px] font-semibold text-primary-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none",
                collapsed ? "size-11 w-full" : "h-10 w-full justify-start px-3",
              )}
            >
              <MessageSquarePlus
                className="size-4 shrink-0 text-[var(--brand)]"
                strokeWidth={2}
                aria-hidden
              />
              {!collapsed && <span>Cuộc trò chuyện mới</span>}
            </button>
          </TooltipTrigger>
          {collapsed && (
            <TooltipContent side="right">Cuộc trò chuyện mới · Ctrl + Shift + O</TooltipContent>
          )}
        </Tooltip>
      </div>

      <nav aria-label="Điều hướng chính" className="px-3 pb-2">
        <ul className="space-y-0.5">
          {NAV.map((n) => {
            const active = n.id === view;
            return (
              <li key={n.id}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      type="button"
                      onClick={() => onViewChange(n.id)}
                      aria-label={n.label}
                      aria-current={active ? "page" : undefined}
                      className={cn(
                        "sidebar-nav-item flex h-10 items-center gap-2.5 overflow-hidden rounded-lg text-[12.5px] transition-colors focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none",
                        collapsed ? "size-11 w-full justify-center" : "w-full px-2",
                        active
                          ? "sidebar-nav-item-active font-medium text-sidebar-accent-foreground"
                          : "text-sidebar-foreground/85 hover:bg-sidebar-accent/60",
                      )}
                    >
                      <span
                        className={cn(
                          "sidebar-nav-icon grid size-7 shrink-0 place-items-center rounded-md",
                          active && "sidebar-nav-icon-active",
                        )}
                      >
                        <n.icon className="size-4" aria-hidden />
                      </span>
                      {!collapsed && <span className="truncate">{n.label}</span>}
                    </button>
                  </TooltipTrigger>
                  {collapsed && <TooltipContent side="right">{n.label}</TooltipContent>}
                </Tooltip>
              </li>
            );
          })}
        </ul>
      </nav>

      <div
        className={cn(
          "scroll-slim min-h-0 flex-1 pt-3",
          collapsed ? "overflow-hidden" : "overflow-y-auto",
        )}
      >
        <ConversationHistory
          conversations={conversations}
          activeId={activeId}
          collapsed={collapsed}
          onSelect={onSelect}
          onRename={onRename}
          onPin={onPin}
          onDelete={onDelete}
        />
      </div>

      <footer className="sidebar-footer shrink-0 border-t border-sidebar-border p-2">
        <div className={cn("flex", collapsed ? "flex-col items-center gap-1" : "flex-col gap-0.5")}>
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                onClick={onOpenSettings}
                aria-label="Cài đặt"
                className={cn(
                  "sidebar-utility-button inline-flex h-9 items-center gap-2.5 rounded-lg px-2.5 text-[12.5px] font-medium text-sidebar-foreground transition-colors hover:bg-sidebar-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none",
                  collapsed ? "size-10 justify-center px-0" : "w-full",
                )}
              >
                <Settings className="size-4 shrink-0" aria-hidden />
                {!collapsed && "Cài đặt"}
              </button>
            </TooltipTrigger>
            {collapsed && <TooltipContent side="right">Cài đặt</TooltipContent>}
          </Tooltip>

          <ThemeToggle theme={theme} onToggle={onToggleTheme} collapsed={collapsed} />

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button
                type="button"
                aria-label="Tài khoản nhóm"
                className={cn(
                  "sidebar-team-card mt-1.5 inline-flex items-center gap-2.5 rounded-[10px] text-left transition-colors hover:bg-sidebar-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none",
                  collapsed ? "size-10 justify-center p-0" : "w-full px-2 py-2",
                )}
              >
                <span className="sidebar-team-icon grid size-7 shrink-0 place-items-center rounded-lg">
                  <Users className="size-3.5" aria-hidden />
                </span>
                {!collapsed && (
                  <>
                    <span className="min-w-0 flex-1 leading-tight">
                      <span className="block truncate text-[12px] font-semibold text-sidebar-foreground">
                        HCMUTE-SHIPCODE
                      </span>
                      <span className="mt-0.5 block truncate text-[10px] text-muted-foreground">
                        Nhóm nghiên cứu
                      </span>
                    </span>
                    <ChevronUp className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
                  </>
                )}
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" side="top" className="w-56">
              <DropdownMenuItem>
                <Users className="size-3.5" aria-hidden /> Thành viên nhóm
              </DropdownMenuItem>
              <DropdownMenuItem onSelect={onOpenSettings}>
                <Settings className="size-3.5" aria-hidden /> Cấu hình hệ thống
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </footer>
    </aside>
  );
}

type PersistentSidebarProps = Omit<
  SidebarProps,
  "collapsed" | "onToggleCollapse" | "inDrawer" | "motionReady"
>;

/** Keeps collapse state local so toggling the rail never re-renders the entire workspace. */
export const PersistentSidebar = memo(function PersistentSidebar(props: PersistentSidebarProps) {
  const [collapsed, setCollapsed, hydrated] = usePersistentState<boolean>(
    "shipcode.sidebar",
    false,
  );
  const [motionReady, setMotionReady] = useState(false);
  const toggleCollapsed = useCallback(() => {
    setCollapsed((value) => !value);
  }, [setCollapsed]);

  useEffect(() => {
    if (!hydrated) return;
    const frame = window.requestAnimationFrame(() => setMotionReady(true));
    return () => window.cancelAnimationFrame(frame);
  }, [hydrated]);

  return (
    <Sidebar
      {...props}
      collapsed={collapsed}
      motionReady={motionReady}
      onToggleCollapse={toggleCollapsed}
    />
  );
});

PersistentSidebar.displayName = "PersistentSidebar";

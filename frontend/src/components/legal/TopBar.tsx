import { ChevronDown, MoreHorizontal, PanelRightOpen, ShieldCheck } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { VietnamFlag } from "./BrandLogo";

interface TopBarProps {
  title: string;
  onOpenEvidence: () => void;
  onRename: () => void;
  onClear: () => void;
  onOpenSettings: () => void;
  evidenceAvailable?: boolean;
  conversationActionsAvailable?: boolean;
  leading?: React.ReactNode;
}

export function TopBar({
  title,
  onOpenEvidence,
  onRename,
  onClear,
  onOpenSettings,
  evidenceAvailable = false,
  conversationActionsAvailable = false,
  leading,
}: TopBarProps) {
  return (
    <header className="vietnam-topbar flex h-14 shrink-0 items-center gap-3 border-b border-border bg-background/85 px-3 backdrop-blur-sm sm:px-5">
      {leading}
      <div className="flex min-w-0 items-center gap-2">
        <h1 className="truncate text-[14px] font-semibold tracking-tight text-foreground">
          {title}
        </h1>
      </div>

      <span className="topbar-vietnam-context inline-flex items-center">
        <VietnamFlag className="w-[21px] shrink-0" decorative />
        <span className="hidden md:inline">Pháp luật Việt Nam</span>
      </span>

      <Tooltip>
        <TooltipTrigger asChild>
          <span className="hidden items-center gap-1.5 text-[11px] text-muted-foreground lg:inline-flex">
            <ShieldCheck className="size-3.5" aria-hidden />
            Đối chiếu theo văn bản gốc
          </span>
        </TooltipTrigger>
        <TooltipContent side="bottom">
          Mọi kết luận đều được đối chiếu với điều, khoản trong kho văn bản đã lập chỉ mục.
        </TooltipContent>
      </Tooltip>

      <div className="ml-auto flex items-center gap-1">
        <button
          type="button"
          onClick={onOpenEvidence}
          disabled={!evidenceAvailable}
          aria-label="Căn cứ pháp lý"
          className="inline-flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1.5 text-[12px] font-medium text-foreground transition-colors hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-45 disabled:hover:bg-transparent"
        >
          <PanelRightOpen className="size-3.5" aria-hidden />
          <span className="hidden sm:inline">Căn cứ pháp lý</span>
        </button>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              type="button"
              aria-label="Tùy chọn khác"
              className="inline-flex size-8 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
            >
              <MoreHorizontal className="size-4" aria-hidden />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-52">
            <DropdownMenuItem disabled={!conversationActionsAvailable} onSelect={onRename}>
              Đổi tên cuộc trò chuyện
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={onOpenSettings}>Cài đặt truy hồi</DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              disabled={!conversationActionsAvailable}
              onSelect={onClear}
              className="text-[var(--flag)]"
            >
              Xóa nội dung hội thoại
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
      <ChevronDown className="hidden size-0" aria-hidden />
    </header>
  );
}
